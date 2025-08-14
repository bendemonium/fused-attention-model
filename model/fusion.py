from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import geoopt

# Reuse Lorentz ops from the hyperbolic module (no circular import there)
from .hyperbolic import minkowski_inner, exp_map, log_map, origin


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


class FusionOnManifold(nn.Module):
    """
    On-manifold fusion of:
      - Euclidean features e_out ∈ R^{B×T×D}
      - Hyperbolic points y_h ∈ H^n ⊂ R^{n+1} as ambient coords (B×T×A), A=n+1

    Steps:
      1) Compute gate α(x)∈(0,1) from [e_out, log_p(y_h)→R^D]
      2) Lift e_out to tangent at anchor p (ManifoldParameter), then exp_p → x_e ∈ H^n
      3) Geodesic interpolation: z_m = Exp_{x_e}( α * Log_{x_e}(y_h) )
      4) Map back to tangent at p: z_t = Log_p(z_m) ∈ R^{A}
      5) Linear proj A→D to return fused Euclidean features (B×T×D)

    Returns:
      z_fused: (B,T,D)
      alpha:   (B,T,1)  (for interpretability)
    """
    def __init__(
        self,
        d_model: int,
        ambient_dim: int,             # A = n+1
        gate_hidden: int = 64,
        fusion_karcher_steps: int = 1 # reserved (we use single-step geodesic interp)
    ):
        super().__init__()
        self.D = d_model
        self.A = ambient_dim

        # ---- Learnable anchor p ∈ H^n (Riemannian parameter) ----
        self.manifold = geoopt.manifolds.Lorentz()
        with torch.no_grad():
            o = origin(self.A - 1, dtype=torch.float32, device=torch.device("cpu"))  # (A,)
            # tiny tangent at origin, then move to a valid point near origin
            eps = torch.zeros(self.A)
            eps[1:] = torch.randn(self.A - 1) * 1e-2
            p0 = exp_map(o, eps)
        self.anchor = geoopt.ManifoldParameter(p0, manifold=self.manifold, requires_grad=True)

        # ---- A few small projections for gating & I/O ----
        # Use log_p(y_h) -> A, then A->D for gating feature
        self.h_log_to_d = nn.Linear(self.A, self.D, bias=True)   # φ_h
        # Lift Euclid to ambient for tangent at p
        self.e_to_ambient = nn.Linear(self.D, self.A, bias=True) # ψ_e
        # Output projection from ambient tangent back to D
        self.out_proj = nn.Linear(self.A, self.D, bias=True)     # W_o

        # Gate α from concatenated Euclid & hyperbolic (in D)
        self.gate = nn.Sequential(
            nn.Linear(2 * self.D, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, e_out: torch.Tensor, y_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        e_out: (B,T,D)  Euclidean stream (post-attn)
        y_h:   (B,T,A)  Hyperbolic stream (ambient coords on H^n)

        returns:
          z_fused: (B,T,D)
          alpha:   (B,T,1)
        """
        B, T, D = e_out.shape
        assert D == self.D, "d_model mismatch"

        # ---- Gate α(e, y) ∈ (0,1) ----
        # Use log-map at anchor p to extract hyperbolic features in ambient tangent, then A->D
        p = self.anchor.to(y_h.dtype).to(y_h.device).view(1, 1, self.A).expand(B, T, self.A)  # (B,T,A)
        y_tan_at_p = log_map(p, y_h)                          # (B,T,A)
        h_feat = self.h_log_to_d(y_tan_at_p)                  # (B,T,D)
        g_in = torch.cat([e_out, h_feat], dim=-1)             # (B,T,2D)
        alpha = self.sigmoid(self.gate(g_in))                 # (B,T,1)

        # ---- Lift Euclidean to H^n via tangent at p ----
        v_raw = self.e_to_ambient(e_out)                      # (B,T,A)
        v_tan = project_tangent(p, v_raw)                     # (B,T,A) ensure <p,v>=0
        x_e = exp_map(p, v_tan)                               # (B,T,A) on H^n

        # ---- Geodesic interpolation toward y_h with weight α ----
        # z_m = Exp_{x_e}( α * Log_{x_e}(y_h) )
        log_xe_y = log_map(x_e, y_h)                          # (B,T,A)
        step = alpha * log_xe_y                                # (B,T,A)
        z_m = exp_map(x_e, step)                              # (B,T,A)

        # ---- Map back to Euclid: log at anchor p then linear A->D ----
        z_tan = log_map(p, z_m)                               # (B,T,A)
        z_fused = self.out_proj(z_tan)                        # (B,T,D)
        return z_fused, alpha