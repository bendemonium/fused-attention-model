from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from more_model_utils import make_padding_mask, LayerNorm, FeedForward, RotaryEmbedding

# ------------------------------
# RoPE (Rotary Positional Embeddings)
# ------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be even"
        half = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))  # (half,)
        t = torch.arange(max_seq_len, dtype=torch.float32)                                # (T,)
        freqs = torch.einsum("t,f->tf", t, inv_freq)                                      # (T, half)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # Duplicate to match interleaving of even/odd
        self.register_buffer("cos_cached", torch.stack([cos, cos], dim=-1).reshape(max_seq_len, dim), persistent=False)
        self.register_buffer("sin_cached", torch.stack([sin, sin], dim=-1).reshape(max_seq_len, dim), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, t, d = q.shape
        cos = self.cos_cached[:t, :].to(q.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh)
        sin = self.sin_cached[:t, :].to(q.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh)

        # split even/odd features
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]
        ce, se = cos[..., ::2], sin[..., ::2]

        q_rot_even = q1 * ce - q2 * se
        q_rot_odd  = q2 * ce + q1 * se
        k_rot_even = k1 * ce - k2 * se
        k_rot_odd  = k2 * ce + k1 * se

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        q_out[..., ::2] = q_rot_even
        q_out[..., 1::2] = q_rot_odd
        k_out[..., ::2] = k_rot_even
        k_out[..., 1::2] = k_rot_odd
        return q_out, k_out


# ------------------------------
# Attention
# ------------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B,H,T,Dh)
        k: torch.Tensor,  # (B,H,T,Dh)
        v: torch.Tensor,  # (B,H,T,Dh)
        mask: Optional[torch.Tensor] = None,  # (B,1,T,T) or broadcastable to that
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dh = q.shape[-1]
        # scores in fp32 for stability
        scores = torch.einsum("bhtd,bhsd->bhts", q.float(), k.float()) / math.sqrt(dh)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        attn = self.dropout(attn)
        ctx = torch.einsum("bhts,bhsd->bhtd", attn.float(), v.float()).to(v.dtype)
        return ctx, attn


class MultiHeadAttentionEuclid(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        rope_max_seq_len: int = 4096,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if use_rope:
            assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_core = ScaledDotProductAttention(attn_dropout=attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=rope_max_seq_len) if use_rope else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(
        self,
        x: torch.Tensor,                      # (B,T,D)
        mask: Optional[torch.Tensor] = None, # (B,1,T,T)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.use_rope:
            q, k = self.rope(q, k)

        ctx, attn = self.attn_core(q, k, v, mask)
        z = self.o_proj(self._combine_heads(ctx))
        z = self.proj_dropout(z)
        return (z, attn) if return_attn else (z, None)


# ------------------------------
# Feed-Forward (GEGLU option ready)
# ------------------------------

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        y = self.fc2(h)
        y = self.dropout(y)
        return y


# ------------------------------
# Encoder Block (Pre-LN)
# ------------------------------

class EncoderBlockEuclid(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_hidden: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        rope_max_seq_len: int = 4096,
        use_rope: bool = True,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln_attn = LayerNorm(d_model, eps=ln_eps)
        self.attn = MultiHeadAttentionEuclid(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=resid_dropout,
            rope_max_seq_len=rope_max_seq_len,
            use_rope=use_rope,
        )
        self.ln_ffn = LayerNorm(d_model, eps=ln_eps)
        self.ffn = FeedForward(d_model=d_model, d_hidden=d_hidden, dropout=ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,                      # (B,T,D)
        attn_mask: Optional[torch.Tensor],   # (B,1,T,T)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        h = self.ln_attn(x)
        a_out, a_weights = self.attn(h, mask=attn_mask, return_attn=return_attn)
        x = x + a_out

        # FFN
        h2 = self.ln_ffn(x)
        f_out = self.ffn(h2)
        x = x + f_out
        return x, a_weights


# ------------------------------
# Encoder Stack
# ------------------------------

@dataclass
class EuclidEncoderConfig:
    vocab_size: int
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_hidden: int = 1024
    dropout: float = 0.1
    attn_dropout: float = 0.0
    rope_max_seq_len: int = 4096
    use_rope: bool = True
    ln_eps: float = 1e-5
    tie_embeddings: bool = True


class EuclideanEncoder(nn.Module):
    """
    Encoder-only stack with token embeddings.
    - Use this alone for Euclidean ablations, or plug it as the Euclid path in mixed models.
    """
    def __init__(self, cfg: EuclidEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            EncoderBlockEuclid(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                d_hidden=cfg.d_hidden,
                attn_dropout=cfg.attn_dropout,
                resid_dropout=cfg.dropout,
                ffn_dropout=cfg.dropout,
                rope_max_seq_len=cfg.rope_max_seq_len,
                use_rope=cfg.use_rope,
                ln_eps=cfg.ln_eps,
            ) for _ in range(cfg.num_layers)
        ])
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,            # (B,T)
        attention_mask: Optional[torch.Tensor] = None,  # (B,T)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.token_embedding(input_ids)  # (B,T,D)
        mask = make_padding_mask(attention_mask) if attention_mask is not None else None

        attn_logs: List[torch.Tensor] = []
        for blk in self.layers:
            x, aw = blk(x, mask, return_attn=return_attn)
            if return_attn:
                attn_logs.append(aw)  # each (B,H,T,T)

        return x, (attn_logs if return_attn else None)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.tie_embeddings:
            # tied weights: logits = H @ E^T
            return hidden @ self.token_embedding.weight.t()
        else:
            return self.lm_head(hidden)