from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from more_model_utils import make_padding_mask, LayerNorm, FeedForward, RotaryEmbedding


# ------------------------------
# Lorentz (hyperboloid) ops
# ------------------------------

EPS = 1e-6
TINY = 1e-15

def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # <x,y>_L = -x0*y0 + sum_{i=1..n} xi*yi
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    # Re-normalize onto H^n: <x,x>_L = -1, x0>0
    xx = minkowski_inner(x, x)
    scale = torch.sqrt(xx.abs().clamp_min(TINY)).unsqueeze(-1)
    x = x / scale
    x0 = x[..., :1].abs()  # ensure time-like positive
    return torch.cat([x0, x[..., 1:]], dim=-1)

def safe_arcosh(z: torch.Tensor) -> torch.Tensor:
    return torch.arccosh(z.clamp_min(1.0 + EPS))

def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # d_H(x,y) = arcosh(-<x,y>_L)
    return safe_arcosh(-minkowski_inner(x, y))

def lorentz_norm_tangent(v: torch.Tensor) -> torch.Tensor:
    vv = minkowski_inner(v, v)
    return torch.sqrt(vv.clamp_min(TINY))

def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # exp_p(v) on H^n
    vn = lorentz_norm_tangent(v).unsqueeze(-1)
    c1 = torch.cosh(vn)
    c2 = torch.sinh(vn) / vn.clamp_min(TINY)
    y = c1 * p + c2 * v
    return project_to_hyperboloid(y)

def log_map(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # log_p(y) on H^n
    alpha = -minkowski_inner(p, y)
    dist = safe_arcosh(alpha).unsqueeze(-1)
    u = y + alpha.unsqueeze(-1) * p
    un = torch.sqrt(minkowski_inner(u, u).clamp_min(TINY)).unsqueeze(-1)
    return (dist / un.clamp_min(TINY)) * u

def tangent_at_origin(u_spatial: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(u_spatial.shape[:-1] + (1,), dtype=u_spatial.dtype, device=u_spatial.device)
    return torch.cat([zeros, u_spatial], dim=-1)

def origin(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    o = torch.zeros(n + 1, dtype=dtype, device=device)
    o[0] = 1.0
    return o

# ------------------------------
# Hyperbolic Attention (Lorentz)
# ------------------------------

class LorentzSelfAttention(nn.Module):
    """
    Lorentz multi-head self-attention.
    Steps:
      - Linear lift to per-head spatial dims (u_q,u_k,u_v ∈ R^{H*n})
      - RoPE on u_q,u_k
      - Append time-like 0 → tangent vectors at origin → expmap at origin → points on H^n
      - Scores = -tau * d_H^2(Q,K); softmax in fp32
      - Aggregate V by one-step Karcher mean around the query point
      - Combine heads by concatenating ambient coords → log-map at per-block anchor p → Linear to d_model
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        spatial_dim: int,            # per-head n; ambient is n+1
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
        self.ambient = spatial_dim + 1
        self.tau = float(tau)
        self.karcher_steps = max(1, karcher_steps)

        # Lift projections (to per-head spatial features)
        self.q_lift = nn.Linear(d_model, num_heads * spatial_dim, bias=True)
        self.k_lift = nn.Linear(d_model, num_heads * spatial_dim, bias=True)
        self.v_lift = nn.Linear(d_model, num_heads * spatial_dim, bias=True)

        # Output projection after log-map at anchor (from H*(n+1) -> d_model)
        self.o_proj = nn.Linear(num_heads * (spatial_dim + 1), d_model, bias=True)

        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_proj = nn.Dropout(proj_dropout)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(spatial_dim, max_seq_len=rope_max_seq_len) if use_rope else None

    def _pack_heads(self, u: torch.Tensor) -> torch.Tensor:
        # (B,T,H*n) -> (B,H,T,n)
        b, t, _ = u.shape
        return u.view(b, t, self.num_heads, self.n).transpose(1, 2)

    def _append_and_lift(self, u: torch.Tensor) -> torch.Tensor:
        # (B,H,T,n) -> (B,H,T,n+1) via exp at origin
        b, h, t, n = u.shape
        zeros = u.new_zeros(b, h, t, 1)
        v = torch.cat([zeros, u], dim=-1)  # tangent vectors at origin: (0, u)
        # origin per head
        o = origin(self.n, dtype=u.dtype, device=u.device)           # (n+1,)
        o_bht = o.view(1, 1, 1, -1).expand(b, h, t, -1)              # (B,H,T,n+1)
        return exp_map(o_bht, v)                                     # points on H^n

    def forward(
        self,
        x: torch.Tensor,                      # (B,T,D)
        anchor_p: torch.Tensor,               # (n+1,) manifold anchor (per block)
        mask: Optional[torch.Tensor] = None,  # (B,1,T,T)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        H, n, A = self.num_heads, self.n, self.ambient

        uq = self._pack_heads(self.q_lift(x))   # (B,H,T,n)
        uk = self._pack_heads(self.k_lift(x))   # (B,H,T,n)
        uv = self._pack_heads(self.v_lift(x))   # (B,H,T,n)

        if self.use_rope:
            uq, uk = self.rope(uq, uk)          # RoPE on spatial pre-lift

        Q = self._append_and_lift(uq)           # (B,H,T,A) on H^n
        K = self._append_and_lift(uk)
        V = self._append_and_lift(uv)

        # Scores = -tau * d_H^2(Q,K)
        # Broadcast to (B,H,T,T,A) for distance; compute in fp32
        Qe = Q.float().unsqueeze(3)              # (B,H,T,1,A)
        Ke = K.float().unsqueeze(2)              # (B,H,1,T,A)
        # d(Qe,Ke) uses Minkowski inner
        d = safe_arcosh_torch(-minkowski_inner(Qe, Ke))  # (B,H,T,T)
        scores = -(self.tau) * (d ** 2)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1).to(V.dtype)  # (B,H,T,T)
        attn = self.dropout_attn(attn)

        # Aggregate V by Fréchet mean with 1 (or k) Karcher steps around the query point Q
        # Start at Q (anchor for iteration)
        Y = Q
        for _ in range(self.karcher_steps):
            # log_{Y}(V_i) weighted by attn over i
            Vexp = V.unsqueeze(2).expand(B, H, T, T, A)          # (B,H,T,T,A)
            Yexp = Y.unsqueeze(3).expand(B, H, T, T, A)
            logv = log_map(Yexp, Vexp)                           # (B,H,T,T,A)
            u = (attn.unsqueeze(-1) * logv).sum(dim=3)           # (B,H,T,A)
            Y = exp_map(Y, u)                                    # (B,H,T,A)

        # Combine heads in ambient: concat along A
        Yc = Y.transpose(1, 2).contiguous().view(B, T, H * A)    # (B,T,H*A)

        # Log-map at the per-block anchor (shared across tokens) and project back to d_model
        p = anchor_p.to(Yc.dtype).to(Yc.device).view(1, 1, A).expand(B, T, A)  # (B,T,A)
        Z_tan = log_map(p, Yc.view(B, T, H, A)).view(B, T, H * A)              # (B,T,H*A)
        Z = self.o_proj(Z_tan)                                                 # (B,T,D)
        Z = self.dropout_proj(Z)
        return (Z, attn if return_attn else None)


def safe_arcosh_torch(z: torch.Tensor) -> torch.Tensor:
    return torch.acosh(z.clamp_min(1.0 + EPS))

# ------------------------------
# Hyperbolic Encoder Block (Pre-LN)
# ------------------------------

class HyperbolicEncoderBlock(nn.Module):
    """
    Pre-LN block with Lorentz attention:
      x -> LN -> LorentzSelfAttention (manifold) -> log-map@anchor -> +res -> LN -> FFN -> +res
    Holds a per-block anchor p_ell ∈ H^n as a ManifoldParameter (learned with Riemannian optimizer).
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

        # Per-block anchor p_ell ∈ H^n as a manifold parameter
        self.manifold = geoopt.manifolds.Lorentz()  # k=1
        # Initialize near origin: sample small spatial vector u, exp_o((0,u))
        n = spatial_dim
        with torch.no_grad():
            u = torch.zeros(n + 1)  # (x0=0, spatial zeros) → exp at origin gives origin; we'll add tiny noise
            u[1:] = torch.randn(n) * 1e-2
            o = origin(n, dtype=torch.float32, device=torch.device("cpu"))
            p0 = exp_map(o, u)  # valid point on H^n
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


# ------------------------------
# Encoder Stack (Hyperbolic-only)
# ------------------------------

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