from __future__ import annotations
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
#    Lorentz utilities
# =========================

EPS = 1e-6
TINY = 1e-12


def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    <x,y>_L = -x0*y0 + sum_{i=1..n} xi*yi
    Shapes:
      x, y: (..., n+1)
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def safe_acosh(z: torch.Tensor) -> torch.Tensor:
    return torch.acosh(z.clamp_min(1.0 + EPS))


def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """
    Re-normalize onto H^n (time-like positive):
      enforce <x,x>_L = -1 and x0 > 0
    """
    xx = minkowski_inner(x, x).abs().clamp_min(TINY)
    scale = torch.sqrt(xx).unsqueeze(-1)
    y = x / scale
    # ensure positive time-like coordinate
    y0 = y[..., :1].abs()
    return torch.cat([y0, y[..., 1:]], dim=-1)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    d_H(x,y) = arcosh(-<x,y>_L), x,y on H^n
    """
    return safe_acosh(-minkowski_inner(x, y))


# =========================
#  Euclidean parametrization
# =========================
#
# From the paper (Sec. 4):
#   p(a,z) = (cosh a,  sinh a * z_hat)           in H^n
#   w(a,z) = (sinh a * ||z||,  cosh a * z)       in T_p H^n
# Hyperplane:
#   H_{z,a} = { x in H^n | cosh(a) <z, x_r> = sinh(a) ||z|| x0 }
# We use the signed function:
#   f(x) = cosh(a) <z, x_r> - sinh(a) ||z|| x0
# f(x)=0 on the plane; sign tells the side.


def _split_time_space(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (..., n+1) -> (x0: (...,1), xr: (...,n))
    """
    return x[..., :1], x[..., 1:]


def params_to_pw(a: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (a,z)  ->  (p,w)
    p in H^n, w in T_p H^n
    Shapes:
      a: (..., 1) or (...,)
      z: (..., n)
    Returns:
      p: (..., n+1)
      w: (..., n+1)
    """
    a = a.squeeze(-1) if a.dim() == z.dim() + 1 else a
    nrm = z.norm(dim=-1, keepdim=True).clamp_min(TINY)
    z_hat = z / nrm

    ca = torch.cosh(a).unsqueeze(-1)  # (...,1)
    sa = torch.sinh(a).unsqueeze(-1)  # (...,1)

    p0 = ca
    pr = sa * z_hat
    p = torch.cat([p0, pr], dim=-1)             # (..., n+1)
    p = project_to_hyperboloid(p)

    w0 = sa * nrm
    wr = ca * z
    w = torch.cat([w0, wr], dim=-1)             # (..., n+1)
    # Guarantee tangent: <p,w>_L ~ 0 (up to numerical)
    return p, w


def pw_to_params(p: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (p,w) -> (a,z), inverse of params_to_pw (up to sign on z-hat when a=0).
    Shapes:
      p: (..., n+1)   with <p,p>_L = -1
      w: (..., n+1)   with <p,w>_L = 0
    Returns:
      a: (..., 1)
      z: (..., n)
    """
    p0, pr = _split_time_space(p)
    w0, wr = _split_time_space(w)

    # a from p0 = cosh a
    ca = p0.clamp_min(1.0 + EPS)
    a = safe_acosh(ca).squeeze(-1)[..., None]  # (...,1)

    # z from wr = cosh a * z  =>  z = wr / cosh a
    z = wr / ca

    # sanity: ||z|| = w0 / sinh a (useful when a is very small)
    sa = torch.sinh(a)
    small = (sa.abs() < 1e-5)
    if small.any():
        # For very small 'a', prefer z = wr / cosh a; already set.
        pass

    return a, z


def hyperplane_signed_value(a: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    f_{a,z}(x) = cosh(a) <z, x_r> - sinh(a) ||z|| x0
    Shapes:
      a: (..., 1)
      z: (..., n)
      x: (..., n+1) points on H^n
    Returns:
      f: (...,) signed value (zero on plane)
    """
    x0, xr = _split_time_space(x)
    nrm = z.norm(dim=-1, keepdim=True).clamp_min(TINY)
    ca = torch.cosh(a)
    sa = torch.sinh(a)
    lhs = (z * xr).sum(dim=-1, keepdim=True)
    rhs = nrm * x0
    f = ca * lhs - sa * rhs
    return f.squeeze(-1)


def hyperplane_distance_like(a: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    A scale-invariant 'distance-like' score:
      d = |f(x)| / (||z|| * cosh(a) + sinh(a))
    This is NOT the exact geodesic distance to the plane (closed form is messy),
    but serves well for margins/losses.
    """
    f = hyperplane_signed_value(a, z, x).abs()
    nrm = z.norm(dim=-1).clamp_min(TINY)
    denom = nrm * torch.cosh(a.squeeze(-1)) + torch.sinh(a.squeeze(-1))
    return f / denom.clamp_min(1e-6)


# =========================
#  Trainable module
# =========================

class LorentzHyperplane(nn.Module):
    """
    Trainable Lorentz hyperplane with Euclidean parameters (a,z).

    f(x) = cosh(a) <z, x_r> - sinh(a) ||z|| x0
    - forward(x)  -> signed value f(x)  (shape (...,))
    - signed_margin(x, y) with y in {+1,-1}
    - hinge / logistic losses

    You can also extract (p,w) by calling .pw()
    """
    def __init__(self, n: int, init_a: float = 0.0, init_scale: float = 0.02):
        """
        Args:
          n: spatial dimension (so ambient is n+1)
          init_a: initial scalar a
          init_scale: std for z init
        """
        super().__init__()
        self.n = n
        a0 = torch.tensor([init_a], dtype=torch.float32)
        z0 = torch.randn(n, dtype=torch.float32) * init_scale
        self.a = nn.Parameter(a0)     # shape (1,)
        self.z = nn.Parameter(z0)     # shape (n,)

    @property
    def dim(self) -> int:
        return self.n

    def pw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (p,w) for current (a,z)."""
        return params_to_pw(self.a, self.z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., n+1) points on H^n
        returns: f(x) (...,)
        """
        a = self.a.view(1, 1).expand(x.shape[:-1] + (1,))
        z = self.z.view(1, -1).expand(x.shape[:-1] + (self.n,))
        return hyperplane_signed_value(a, z, x)

    def signed_margin(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y in {+1,-1} (same broadcastable shape as f(x))
        returns y * f(x)
        """
        f = self.forward(x)
        return y.to(f.dtype) * f

    def hinge_loss(self, x: torch.Tensor, y: torch.Tensor, margin: float = 1.0, reduction: str = "mean") -> torch.Tensor:
        sm = self.signed_margin(x, y)
        loss = F.relu(margin - sm)
        return loss.mean() if reduction == "mean" else loss.sum()

    def logistic_loss(self, x: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        sm = self.signed_margin(x, y)
        loss = F.softplus(-sm)  # log(1+exp(-sm))
        return loss.mean() if reduction == "mean" else loss.sum()

    def l2_reg(self, weight: float = 1e-4) -> torch.Tensor:
        """Simple regularizer on z and a to discourage blow-ups."""
        return weight * (self.z.square().sum() + self.a.square())

    def clamp_params(self, max_abs_a: float = 10.0, max_norm_z: float = 1e3):
        """
        Optional runtime guard to prevent extreme values.
        """
        with torch.no_grad():
            self.a.clamp_(-max_abs_a, max_abs_a)
            zn = self.z.norm().clamp_min(EPS)
            if zn > max_norm_z:
                self.z.mul_(max_norm_z / zn)


# =========================
#  Batch helpers
# =========================

def batch_signed_value(
    a: torch.Tensor,         # (B,1) or (B,)
    z: torch.Tensor,         # (B,n)
    x: torch.Tensor,         # (B,*,n+1)
) -> torch.Tensor:
    """
    Per-batch hyperplanes applied to per-batch (and possibly sequence) of points.
    Returns f: (B,*) broadcasting over the middle dims of x.
    """
    B = z.shape[0]
    a = a.view(B, 1)
    # expand z to x middle dims
    mid = x.shape[1:-1]
    a_exp = a.view(B, *([1] * len(mid)), 1).expand(B, *mid, 1)
    z_exp = z.view(B, *([1] * len(mid)), z.shape[-1]).expand(B, *mid, z.shape[-1])
    return hyperplane_signed_value(a_exp, z_exp, x)


def batch_hinge_loss(
    a: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    f = batch_signed_value(a, z, x)
    loss = F.relu(margin - y.to(f.dtype) * f)
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


# =========================
#  Example usage (doc)
# =========================
#
# hp = LorentzHyperplane(n=32)
# pts: (B,T,33) on H^32
# f = hp(pts)             # (B,T) signed values
# loss = hp.hinge_loss(pts, y, margin=1.0)
# reg  = hp.l2_reg(1e-5)
# (p, w) = hp.pw()
#
# To get (a,z) back from (p,w):
# a_rec, z_rec = pw_to_params(p, w)
#
# You can plug this module into gating/decision layers or for diagnostics,
# and it plays nicely with the Euclidean parametrization in hyperbolic.py.