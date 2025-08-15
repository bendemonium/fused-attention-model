# model/fusion.py

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the same numerically-stable Lorentz ops that your new hyperbolic.py provides.
# If this path differs in your tree, adjust the import.
from .hyperbolic import (
    EPS, TINY,
    minkowski_inner, safe_acosh,              # (we’ll use safe_acosh via a local alias too)
)

# ---------- local safe acosh (alias to keep file self-contained if needed) ----------
def _safe_acosh(z: torch.Tensor) -> torch.Tensor:
    return torch.acosh(z.clamp_min(1.0 + EPS))


# ---------- basic Lorentz helpers (kept here to avoid circular imports) ----------
def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """
    Re-normalize onto H^n with positive time-like coordinate.
      enforce <x,x>_L = -1 and x0 > 0
    """
    xx = minkowski_inner(x, x).abs().clamp_min(TINY)
    scale = torch.sqrt(xx).unsqueeze(-1)
    y = x / scale
    y0 = y[..., :1].abs()
    return torch.cat([y0, y[..., 1:]], dim=-1)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _safe_acosh(-minkowski_inner(x, y))


def log_map(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Logarithm map on H^n at p toward y (ambient Lorentz coords).
    Shapes: (..., A)
    """
    alpha = -minkowski_inner(p, y)                    # (...,)
    dist = _safe_acosh(alpha).unsqueeze(-1)           # (...,1)
    u = y + alpha.unsqueeze(-1) * p                   # (...,A)
    un = torch.sqrt(minkowski_inner(u, u).clamp_min(TINY)).unsqueeze(-1)
    return (dist / un.clamp_min(TINY)) * u            # (...,A)


def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on H^n at p for tangent v (ambient Lorentz coords).
    Shapes: (..., A)
    """
    vn = torch.sqrt(minkowski_inner(v, v).clamp_min(TINY)).unsqueeze(-1)
    c1 = torch.cosh(vn)
    c2 = torch.sinh(vn) / vn.clamp_min(TINY)
    y = c1 * p + c2 * v
    return project_to_hyperboloid(y)


def project_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project v to T_p H^n:  v_t = v + <p,v>_L * p
    """
    coeff = minkowski_inner(p, v).unsqueeze(-1)
    return v + coeff * p


# ---------- Euclidean parametrization of the anchor p(a,z) ----------
def anchor_from_params(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Build anchor point p ∈ H^n from euclidean params (a,z)
      p = (cosh a,  sinh a * z_hat)
    a: (1,) learnable scalar
    z: (n,) learnable spatial vector
    returns p: (A,) with A=n+1
    """
    nrm = z.norm().clamp_min(TINY)
    z_hat = z / nrm
    ca = torch.cosh(a)
    sa = torch.sinh(a)
    p0 = ca.view(1)
    pr = (sa * z_hat).view(-1)
    p = torch.cat([p0, pr], dim=0)
    return project_to_hyperboloid(p)


# ==================================================================================
#                                   Fusion
# ==================================================================================

class FusionOnManifold(nn.Module):
    """
    On-manifold fusion between:
      - e_out: Euclidean features R^{B×T×D}
      - y_h:   Lorentz ambient points on H^n, R^{B×T×A} with A=n+1

    Steps
      1) Anchor p from (a,z) euclidean params (shared across time; broadcast across batch)
      2) Gate α = σ(MLP([e_out, φ_h(log_p(y_h))])) with φ_h: A→D
      3) Lift e_out to tangent at p: ψ_e: D→A, then T_p via project_tangent, then exp_p to H^n
      4) Geodesic step toward y_h: z_m = Exp_{x_e}( α * Log_{x_e}(y_h) )
      5) Log back at p and linearly project A→D

    Returns:
      z_fused: (B,T,D)
      alpha:   (B,T,1)
    """

    def __init__(
        self,
        d_model: int,
        ambient_dim: int,          # A = n+1
        gate_hidden: int = 64,
        fusion_karcher_steps: int = 1,   # reserved (kept for API compat)
    ):
        super().__init__()
        self.D = d_model
        self.A = ambient_dim
        self.n = ambient_dim - 1

        # ----- learnable euclidean params for anchor -----
        self.a = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))           # scalar
        self.z = nn.Parameter(torch.randn(self.n, dtype=torch.float32) * 1e-2)    # spatial

        # ----- projections for gating / lifting / output -----
        self.h_log_to_d = nn.Linear(self.A, self.D, bias=True)   # φ_h
        self.e_to_ambient = nn.Linear(self.D, self.A, bias=True) # ψ_e
        self.out_proj = nn.Linear(self.A, self.D, bias=True)     # W_o

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.D, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # for debugging / tracing
        self.last_alpha: Optional[torch.Tensor] = None

    # -------- convenience accessor for modules outside (e.g., passing anchor to attention) --------
    def anchor_point(self, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns p ∈ H^n as (A,) or broadcast to 'like' leading dims ending in A.
        """
        p = anchor_from_params(self.a, self.z)  # (A,)
        if like is None:
            return p
        # Broadcast to like[..., A]
        shape = like.shape[:-1] + (self.A,)
        return p.view(*([1] * (len(shape) - 1)), self.A).expand(shape)

    def forward(self, e_out: torch.Tensor, y_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        e_out: (B,T,D)
        y_h:   (B,T,A)  (H^n points)
        returns:
          z_fused: (B,T,D)
          alpha:   (B,T,1)
        """
        assert e_out.dim() == 3 and y_h.dim() == 3, "Expect (B,T,D) and (B,T,A)"
        B, T, D = e_out.shape
        assert y_h.shape[-1] == self.A, "ambient_dim mismatch"

        # ----- 1) anchor p (broadcast over B,T) -----
        p = self.anchor_point(like=y_h)                   # (B,T,A)

        # ----- 2) gate α(e, y) -----
        y_tan_at_p = log_map(p, y_h)                      # (B,T,A)
        h_feat = self.h_log_to_d(y_tan_at_p)              # (B,T,D)
        g_in = torch.cat([e_out, h_feat], dim=-1)         # (B,T,2D)

        # numeric guard: keep pre-activations tame
        g_in = g_in.clamp(min=-50.0, max=50.0)
        alpha = self.sigmoid(self.gate_mlp(g_in))         # (B,T,1)

        # ----- 3) lift e_out to H^n through tangent at p -----
        v_raw = self.e_to_ambient(e_out)                  # (B,T,A)
        v_tan = project_tangent(p, v_raw)                 # (B,T,A)
        x_e = exp_map(p, v_tan)                           # (B,T,A)

        # ----- 4) geodesic interpolation toward y_h with α -----
        log_xe_to_y = log_map(x_e, y_h)                   # (B,T,A)
        step = alpha * log_xe_to_y                         # (B,T,A)
        z_m = exp_map(x_e, step)                          # (B,T,A)

        # ----- 5) map back to ℝ^D -----
        z_tan = log_map(p, z_m)                           # (B,T,A)
        z_fused = self.out_proj(z_tan)                    # (B,T,D)

        self.last_alpha = alpha.detach()
        return z_fused, alpha

    # -------- optional runtime clamps to avoid parameter blow-ups --------
    def clamp_params(self, max_abs_a: float = 10.0, max_norm_z: float = 1e3):
        with torch.no_grad():
            self.a.clamp_(-max_abs_a, max_abs_a)
            zn = self.z.norm().clamp_min(EPS)
            if zn > max_norm_z:
                self.z.mul_(max_norm_z / zn)