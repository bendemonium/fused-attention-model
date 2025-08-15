from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from .more_model_utils import make_padding_mask, LayerNorm, FeedForward, RotaryEmbedding

# ------------------------------
# Numeric constants
# ------------------------------
EPS = 1e-6         # general small epsilon
TINY = 1e-15       # stricter tiny
MAX_R = 18.0       # clamp ‖z‖ before cosh/sinh to avoid overflow in fp32


# ------------------------------
# Lorentz (hyperboloid) ops
# ------------------------------

def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """<x,y>_L = -x0*y0 + sum_{i=1..n} xi*yi, broadcasting over leading dims."""
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

def safe_acosh(z: torch.Tensor) -> torch.Tensor:
    return torch.acosh(z.clamp_min(1.0 + EPS))

def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """d_H(x,y) = arcosh(-<x,y>_L)"""
    return safe_acosh(-minkowski_inner(x, y))

def lorentz_norm_tangent(v: torch.Tensor) -> torch.Tensor:
    """‖v‖_L for tangent vectors (positive)."""
    vv = minkowski_inner(v, v)
    return torch.sqrt(vv.clamp_min(TINY))

def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """
    Re-normalize onto H^n: <x,x>_L = -1, x0>0
    (Not a true orthogonal projection; used only as a last-resort safety re-normalizer.)
    """
    xx = minkowski_inner(x, x).abs().clamp_min(TINY)  # positive
    scale = torch.sqrt(xx).unsqueeze(-1)
    y = x / scale
    y0 = y[..., :1].abs()
    return torch.cat([y0, y[..., 1:]], dim=-1)

def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on H^n (Lorentz model).
    Assumes v ∈ T_p H^n (i.e., <p,v>_L = 0). No-op projection for speed; upstream ensures tangency.
    """
    dtype_orig = v.dtype
    p32, v32 = p.float(), v.float()
    vn = lorentz_norm_tangent(v32).unsqueeze(-1)      # (...,1)
    c1 = torch.cosh(vn)
    c2 = torch.sinh(vn) / vn.clamp_min(TINY)
    y = c1 * p32 + c2 * v32
    y = project_to_hyperboloid(y)
    return y.to(dtype_orig)

def log_map(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Logarithm map on H^n (Lorentz model)."""
    dtype_orig = y.dtype
    p32, y32 = p.float(), y.float()
    alpha = -minkowski_inner(p32, y32)                         # (...,)
    dist = safe_acosh(alpha).unsqueeze(-1)                     # (...,1)
    u = y32 + alpha.unsqueeze(-1) * p32                        # (...,A)
    un = torch.sqrt(minkowski_inner(u, u).clamp_min(TINY)).unsqueeze(-1)
    v = (dist / un.clamp_min(TINY)) * u
    return v.to(dtype_orig)

def tangent_at_origin(u_spatial: torch.Tensor) -> torch.Tensor:
    """Build tangent vector at origin from spatial part u (append 0 time-like)."""
    zeros = torch.zeros(u_spatial.shape[:-1] + (1,), dtype=u_spatial.dtype, device=u_spatial.device)
    return torch.cat([zeros, u_spatial], dim=-1)

def origin(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Lorentz origin \bar{0} = (1, 0, ..., 0) ∈ R^{n+1}."""
    o = torch.zeros(n + 1, dtype=dtype, device=device)
    o[0] = 1.0
    return o


# ------------------------------
# Euclidean parametrization  (Section 4 of the paper)
# ------------------------------

def FL(z: torch.Tensor) -> torch.Tensor:
    """
    Map Euclidean parameter z ∈ R^n to a Lorentz point y ∈ H^n:
      r = ||z||,  s = z / r
      y0 = cosh(r),  y_spatial = sinh(r) * s
    Safe for r≈0 and clamped r to avoid overflow.
    Supports arbitrary leading dims; last dim is n.
    """
    z32 = z.float()
    r = torch.linalg.norm(z32, dim=-1, keepdim=True)                     # (...,1)
    r_safe = r.clamp_min(1e-12).clamp_max(MAX_R)
    s = z32 / r_safe                                                     # (...,n)
    y0 = torch.cosh(r_safe)                                              # (...,1)
    ys = torch.sinh(r_safe) * s                                          # (...,n)
    y = torch.cat([y0, ys], dim=-1)                                      # (...,n+1)
    return project_to_hyperboloid(y).to(z.dtype)


# ------------------------------
# Stable masked softmax
# ------------------------------

def masked_rowwise_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Row-stable masked softmax:
      - supports mask as boolean (True = keep)
      - returns 0 on fully-masked rows (prevents NaN)
    """
    if mask is None:
        return torch.softmax(scores, dim=dim)
    # Align dtypes/devices
    scores_f = scores.float()
    mask_f = mask.to(dtype=scores_f.dtype)
    # Replace masked positions with -inf in a numerically stable way
    neg_inf = torch.finfo(scores_f.dtype).min
    masked_scores = scores_f.masked_fill(~mask, neg_inf)
    # Subtract max per row
    row_max = masked_scores.max(dim=dim, keepdim=True).values
    exps = torch.exp(masked_scores - row_max) * mask_f
    denom = exps.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    out = (exps / denom)
    return out.to(scores.dtype)


# ------------------------------
# Lorentz Self-Attention (with Euclidean parametrization)
# ------------------------------

class LorentzSelfAttention(nn.Module):
    """
    Hyperbolic (Lorentz) multi-head self-attention with Euclidean parametrization of Q/K/V:
      - Predict z_q, z_k, z_v ∈ R^{H*n} → per-head split → map by FL(z) to H^n
      - Scores: -tau * d_H^2(Q, K), computed in fp32
      - Weights: row-stable masked softmax (no NaN from fully-masked rows)
      - Values: aggregated on manifold by Karcher steps around the query point
      - Output:
          * if return_points: per-token manifold representative Y_token ∈ R^{B×T×(n+1)}
          * else: log-map at anchor to tangent and project to d_model
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        spatial_dim: int,            # per-head n; ambient A=n+1
        tau: float = 1.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        rope_max_seq_len: int = 4096,
        use_rope: bool = True,
        karcher_steps: int = 1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        if use_rope:
            assert spatial_dim % 2 == 0, "RoPE requires even spatial_dim per head"

        self.d_model = d_model
        self.num_heads = num_heads
        self.n = spatial_dim
        self.A = spatial_dim + 1
        self.tau = float(tau)
        self.karcher_steps = max(1, int(karcher_steps))

        # Linear lifts to Euclidean z for each head (size n per head)
        self.q_lift = nn.Linear(d_model, num_heads * spatial_dim, bias=True)
        self.k_lift = nn.Linear(d_model, num_heads * spatial_dim, bias=True)
        self.v_lift = nn.Linear(d_model, num_heads * spatial_dim, bias=True)

        # Output projection after log-map@anchor: (H*(n+1)) -> d_model
        self.o_proj = nn.Linear(num_heads * (spatial_dim + 1), d_model, bias=True)

        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_proj = nn.Dropout(proj_dropout)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(spatial_dim, max_seq_len=rope_max_seq_len) if use_rope else None

        # For debugging / tracing
        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_points_token: Optional[torch.Tensor] = None  # (B,T,A)

    # ---- helpers ----
    def _split_heads(self, u: torch.Tensor) -> torch.Tensor:
        # (B,T,H*n) -> (B,H,T,n)
        B, T, _ = u.shape
        return u.view(B, T, self.num_heads, self.n).transpose(1, 2).contiguous()

    def _pairwise_distance(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Q,K: (B,H,T,A)
        returns: d(Q_i, K_j) for each i,j -> (B,H,T,T)
        """
        Qe = Q.float().unsqueeze(3)  # (B,H,T,1,A)
        Ke = K.float().unsqueeze(2)  # (B,H,1,T,A)
        ip = -minkowski_inner(Qe, Ke).clamp_min(1.0 + EPS)  # (B,H,T,T)
        return safe_acosh(ip)  # (B,H,T,T)

    def forward(
        self,
        x: torch.Tensor,                              # (B,T,D)
        anchor_p: Optional[torch.Tensor] = None,      # (A,) or (B,T,A); if None, origin
        mask: Optional[torch.Tensor] = None,          # (B,1,T,T) with True=keep
        return_attn: bool = False,
        return_points: bool = False,
    ):
        B, T, D = x.shape
        H, n, A = self.num_heads, self.n, self.A

        # 1) Lift to Euclidean per-head z and (optionally) apply RoPE on z for Q/K
        zq = self._split_heads(self.q_lift(x))  # (B,H,T,n)
        zk = self._split_heads(self.k_lift(x))  # (B,H,T,n)
        zv = self._split_heads(self.v_lift(x))  # (B,H,T,n)

        if self.use_rope:
            zq, zk = self.rope(zq, zk)         # same shape

        # 2) Map to H^n via Euclidean parametrization (fp32 inside FL)
        Q = FL(zq)  # (B,H,T,A)
        K = FL(zk)  # (B,H,T,A)
        V = FL(zv)  # (B,H,T,A)

        # 3) Scores: -tau * d^2; safe, fp32
        d = self._pairwise_distance(Q, K)                 # (B,H,T,T)
        scores = -(self.tau) * (d ** 2)                   # (B,H,T,T)

        # 4) Mask + row-stable softmax
        m = None
        if mask is not None:
            # mask: (B,1,T,T) -> (B,H,T,T)
            m = mask.expand(B, H, T, T)
        attn = masked_rowwise_softmax(scores, m, dim=-1)  # (B,H,T,T)
        attn = self.dropout_attn(attn)
        self.last_attention_weights = attn

        # 5) Aggregate on manifold via Karcher steps around the query point
        Y = Q  # initial center
        for _ in range(self.karcher_steps):
            Vexp = V.unsqueeze(2).expand(B, H, T, T, A)   # (B,H,T,T,A)
            Yexp = Y.unsqueeze(3).expand(B, H, T, T, A)
            logv = log_map(Yexp, Vexp)                    # (B,H,T,T,A)
            u = (attn.unsqueeze(-1) * logv).sum(dim=3)    # (B,H,T,A) tangent at Y
            Y = exp_map(Y, u)                             # (B,H,T,A)

        # Token-level representative (average across heads)
        Y_tok = Y.mean(dim=1)  # (B,T,A)
        self.last_points_token = Y_tok

        if return_points and not return_attn:
            return Y_tok
        if return_points and return_attn:
            return (Y_tok, attn)

        # 6) Otherwise: log-map@anchor and project to d_model
        # Prepare anchor
        if anchor_p is None:
            p0 = origin(n, dtype=Y.dtype, device=Y.device).view(1, 1, A).expand(B, T, A)  # (B,T,A)
        else:
            p0 = anchor_p.to(dtype=Y.dtype, device=Y.device)
            if p0.dim() == 1:
                p0 = p0.view(1, 1, A).expand(B, T, A)  # (B,T,A)

        # Log-map per head
        Y_bTHA = Y.transpose(1, 2).contiguous()         # (B,T,H,A)
        p0_bTHA = p0.unsqueeze(2).expand(B, T, H, A)    # (B,T,H,A)
        Z_tan = log_map(p0_bTHA, Y_bTHA)                # (B,T,H,A)
        Z_tan = Z_tan.reshape(B, T, H * A)              # (B,T,H*A)
        Z = self.o_proj(Z_tan)                          # (B,T,D)
        Z = self.dropout_proj(Z)
        return (Z, attn if return_attn else None)


# ------------------------------
# Hyperbolic Encoder Block / Stack (optional path)
# ------------------------------

class HyperbolicEncoderBlock(nn.Module):
    """
    Pre-LN block with Lorentz attention:
      x -> LN -> LorentzSelfAttention (Euclid-param QKV) -> log-map@anchor -> +res -> LN -> FFN -> +res
    Holds a per-block anchor p_ell ∈ H^n as a ManifoldParameter (Riemannian-optimized).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        spatial_dim: int,
        d_hidden: int,
        tau: float = 1.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        rope_max_seq_len: int = 4096,
        use_rope: bool = True,
        ln_eps: float = 1e-5,
        karcher_steps: int = 1,
    ):
        super().__init__()
        self.ln_attn = LayerNorm(d_model, eps=ln_eps)

        self.attn = LorentzSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
            tau=tau,
            attn_dropout=attn_dropout,
            proj_dropout=resid_dropout,
            rope_max_seq_len=rope_max_seq_len,
            use_rope=use_rope,
            karcher_steps=karcher_steps,
        )

        self.ln_ffn = LayerNorm(d_model, eps=ln_eps)
        self.ffn = FeedForward(d_model=d_model, d_hidden=d_hidden, dropout=ffn_dropout)

        # Per-block anchor p_ell ∈ H^n as a manifold parameter (near origin)
        self.manifold = geoopt.manifolds.Lorentz()
        n = spatial_dim
        with torch.no_grad():
            o = origin(n, dtype=torch.float32, device=torch.device("cpu"))
            # tiny tangent in spatial directions
            u = torch.zeros(n + 1)
            u[1:] = torch.randn(n) * 1e-2
            p0 = exp_map(o, u)
        self.anchor = geoopt.ManifoldParameter(p0, manifold=self.manifold, requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,                      # (B,T,D)
        attn_mask: Optional[torch.Tensor],   # (B,1,T,T)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], geoopt.ManifoldParameter]:
        h = self.ln_attn(x)
        z, a_weights = self.attn(h, anchor_p=self.anchor, mask=attn_mask, return_attn=return_attn)
        x = x + z
        y = self.ln_ffn(x)
        y = self.ffn(y)
        x = x + y
        return x, (a_weights if return_attn else None), self.anchor


@dataclass
class HyperbolicEncoderConfig:
    vocab_size: int
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    spatial_dim: int = 32         # per-head n (ambient n+1)
    d_hidden: int = 1024
    tau: float = 1.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    rope_max_seq_len: int = 4096
    use_rope: bool = True
    ln_eps: float = 1e-5
    karcher_steps: int = 1
    tie_embeddings: bool = True


class HyperbolicEncoder(nn.Module):
    """
    Encoder-only stack with token embeddings.
    Hyperbolic attention inside each block; anchors are ManifoldParameters.
    """
    def __init__(self, cfg: HyperbolicEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            HyperbolicEncoderBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                spatial_dim=cfg.spatial_dim,
                d_hidden=cfg.d_hidden,
                tau=cfg.tau,
                attn_dropout=cfg.attn_dropout,
                resid_dropout=cfg.dropout,
                ffn_dropout=cfg.dropout,
                rope_max_seq_len=cfg.rope_max_seq_len,
                use_rope=cfg.use_rope,
                ln_eps=cfg.ln_eps,
                karcher_steps=cfg.karcher_steps,
            ) for _ in range(cfg.num_layers)
        ])
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,            # (B,T)
        attention_mask: Optional[torch.Tensor] = None,  # (B,T)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], List[geoopt.ManifoldParameter]]:
        x = self.token_embedding(input_ids)  # (B,T,D)
        mask = make_padding_mask(attention_mask) if attention_mask is not None else None

        attn_logs: List[torch.Tensor] = []
        anchors: List[geoopt.ManifoldParameter] = []
        for blk in self.layers:
            x, aw, anchor = blk(x, mask, return_attn=return_attn)
            anchors.append(anchor)
            if return_attn:
                attn_logs.append(aw)  # each (B,H,T,T)

        return x, (attn_logs if return_attn else None), anchors

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.tie_embeddings:
            return hidden @ self.token_embedding.weight.t()
        else:
            return self.lm_head(hidden)