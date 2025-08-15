# model/fusion.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import geoopt

# Reuse Lorentz ops from the hyperbolic module (no circular import there)
from .hyperbolic import minkowski_inner, exp_map, log_map, origin, fl_from_z, hyperplane_gate_score

# ---------- numeric safety constants (match hyperbolic.py style) ----------
EPS = 1e-6
TINY = 1e-15
MAX_STEP_NORM = 20.0   # cap tangent step size before exp_map


def project_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project arbitrary ambient vector v onto the tangent space T_p H^n:
      T_p H^n = { u : <p,u>_L = 0 }  with Minkowski inner product.
    Using: u = v + <p,v>_L * p
    Shapes:
      p: (..., A)
      v: (..., A)
    """
    coeff = minkowski_inner(p, v).unsqueeze(-1)   # (...,1)
    return v + coeff * p


def lorentz_norm_tangent(v: torch.Tensor) -> torch.Tensor:
    """||v||_L in tangent (uses Minkowski inner)."""
    vv = minkowski_inner(v, v)
    return torch.sqrt(vv.clamp_min(TINY))


class FusionOnManifold(nn.Module):
    """
    Geometry-aware fusion (Euclid + Hyperbolic) with Euclidean parametrization:

      Inputs:
        e_out : (B,T,D)  Euclidean stream (post-attn)
        y_h   : (B,T,A)  Hyperbolic stream (ambient Lorentz coords), A = n+1

      Parameters (learned, Euclidean):
        - Anchor z_anchor ∈ R^n  → p = FL(z_anchor) ∈ H^n   (for log/exp maps)
        - Gate   z_gate  ∈ R^n, a_gate ∈ R   → α = σ(γ * s(x)) with
              s(x) = cosh(a)<z, x_r> - sinh(a)||z|| x0     (Eq. in §4.2)
        - Linear lifts/proj: e_to_ambient: D→A, out_proj: A→D

      Steps:
        1) α(x) from hyperbolic hyperplane score at y_h
        2) Lift e_out to T_p via e_to_ambient, project to tangent, then Exp_p -> x_e
        3) Geodesic interp:  z_m = Exp_{x_e}( α * Log_{x_e}(y_h) )
        4) Back to Euclid:  z_tan = Log_p(z_m),  z_fused = out_proj(z_tan)
    """
    def __init__(self,
                 d_model: int,
                 ambient_dim: int,          # A = n+1
                 gate_hidden: int = 64,     # kept for API compatibility (unused)
                 fusion_karcher_steps: int = 1,  # kept for API compatibility
                 max_step_norm: float = 5.0,
                 max_rad: float = 10.0,
                 score_scale_init: float = 1.0):
        super().__init__()
        self.D = d_model
        self.A = ambient_dim
        self.n = ambient_dim - 1
        self.max_step = max_step_norm
        self.max_rad = max_rad

        # ----- Euclidean parametrization for anchor & gate -----
        self.z_anchor = nn.Parameter(torch.zeros(self.n))           # R^n
        nn.init.normal_(self.z_anchor, mean=0.0, std=1e-2)

        self.z_gate   = nn.Parameter(torch.zeros(self.n))           # R^n (direction)
        nn.init.normal_(self.z_gate, mean=0.0, std=1e-2)
        self.a_gate   = nn.Parameter(torch.tensor(0.0))             # scalar offset
        self.score_scale = nn.Parameter(torch.tensor(float(score_scale_init)))  # γ

        # ----- Linear lifts/projections -----
        self.e_to_ambient = nn.Linear(self.D, self.A, bias=True)    # D -> A
        self.out_proj      = nn.Linear(self.A, self.D, bias=True)   # A -> D

        # Dropouts (match prior API; keep off by default)
        self.dropout_out = nn.Dropout(p=0.0)

    def forward(self, e_out: torch.Tensor, y_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        e_out: (B,T,D), y_h: (B,T,A) with A=n+1 on H^n (Lorentz coords)
        returns (z_fused: (B,T,D), alpha: (B,T,1))
        """
        B, T, D = e_out.shape
        assert D == self.D and y_h.size(-1) == self.A

        # ----- 1) Hyperplane gate α(y_h) ∈ (0,1) -----
        # score s(x) = cosh(a)<z,x_r> - sinh(a)||z|| x0
        s = hyperplane_gate_score(y_h, self.z_gate, self.a_gate, max_a=self.max_rad)  # (B,T,1)
        alpha = torch.sigmoid(self.score_scale * s).clamp(1e-6, 1.0 - 1e-6)           # (B,T,1)

        # ----- 2) Build anchor p from Euclidean z_anchor via FL -----
        p = fl_from_z(self.z_anchor, max_rad=self.max_rad)                  # (A,)
        p = p.to(y_h.dtype).to(y_h.device).view(1, 1, self.A).expand(B, T, self.A)

        # Lift Euclidean stream to tangent at p and Exp_p
        v_raw = self.e_to_ambient(e_out)                                    # (B,T,A)
        v_tan = project_tangent(p, v_raw)                                   # <p,v>=0
        # Trust region on tangent step
        v_norm = torch.sqrt(torch.clamp(torch.abs(minkowski_inner(v_tan, v_tan)), min=1e-12)).unsqueeze(-1)
        scale = (self.max_step / v_norm).clamp_max(1.0)
        v_tan = v_tan * scale
        x_e = exp_map(p, v_tan)                                             # (B,T,A)

        # ----- 3) Geodesic interpolation: Exp_{x_e}( α * Log_{x_e}(y_h) ) -----
        log_xe_y = log_map(x_e, y_h)                                        # (B,T,A)
        # cap the tangent move at max_step too
        u_norm = torch.sqrt(torch.clamp(torch.abs(minkowski_inner(log_xe_y, log_xe_y)), min=1e-12)).unsqueeze(-1)
        scale_u = (self.max_step / u_norm).clamp_max(1.0)
        step = alpha * (log_xe_y * scale_u)
        z_m = exp_map(x_e, step)                                            # (B,T,A)

        # ----- 4) Back to Euclid: Log_p then A->D -----
        z_tan = log_map(p, z_m)                                             # (B,T,A)
        z_fused = self.out_proj(z_tan)                                      # (B,T,D)
        z_fused = self.dropout_out(z_fused)

        return z_fused, alpha