from __future__ import annotations
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .more_model_utils import LayerNorm, FeedForward, RotaryEmbedding, make_padding_mask


# ---------- stable masked softmax (prevents NaNs on fully-masked rows) ----------

def masked_rowwise_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Row-stable masked softmax:
      - mask should be boolean with True = keep, False = mask out
      - fully-masked rows return all zeros (not NaN)
    """
    if mask is None:
        return torch.softmax(scores, dim=dim)
    scores_f = scores.float()
    neg_inf = torch.finfo(scores_f.dtype).min
    masked = scores_f.masked_fill(~mask, neg_inf)
    row_max = masked.max(dim=dim, keepdim=True).values
    exps = torch.exp(masked - row_max) * mask.to(dtype=scores_f.dtype)
    denom = exps.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    out = exps / denom
    return out.to(dtype=scores.dtype)


# --------------------------------- MHA (Euclidean) ---------------------------------

class MultiHeadAttentionEuclid(nn.Module):
    """
    Standard multi-head self-attention (Euclidean) with improved masking.
    Returns:
      - if return_attn=False: Tensor (B,T,D)
      - if return_attn=True: Tuple[Tensor (B,T,D), Tensor (B,H,T,T)]
    Mask shape expected: (B,1,T,T) with True=keep.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_rope: bool = False,
        rope_max_seq_len: int = 4096,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        # RoPE if requested
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=rope_max_seq_len)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=rope_max_seq_len) if use_rope else None

        # for optional debugging
        self.last_attention_weights: Optional[torch.Tensor] = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) -> (B,H,T,head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,                             # (B,T,D)
        mask: Optional[torch.Tensor] = None,         # (B,1,T,T) True=keep
        return_attn: bool = False,
    ):
        B, T, D = x.shape

        q = self._shape(self.q_proj(x))   # (B,H,T,hd)
        k = self._shape(self.k_proj(x))   # (B,H,T,hd)
        v = self._shape(self.v_proj(x))   # (B,H,T,hd)

        if self.use_rope:
            q, k = self.rope(q, k)        # RoPE applied over last dim

        # scaled dot-prod
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        # expand mask to (B,H,T,T) if provided
        m = mask.expand(B, self.num_heads, T, T) if mask is not None else None
        attn = masked_rowwise_softmax(scores, m, dim=-1)            # (B,H,T,T)
        attn = self.attn_drop(attn)
        self.last_attention_weights = attn

        y = torch.matmul(attn, v)                                   # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, D)            # (B,T,D)
        y = self.out_proj(y)
        y = self.proj_drop(y)

        if return_attn:
            return y, attn
        return y


# ------------------------------- Transformer encoder (Euclid) -------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block: x -> LN -> MHA -> +res -> LN -> FFN -> +res
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_rope: bool = False,
        rope_max_seq_len: int = 4096,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln_attn = LayerNorm(d_model, eps=ln_eps)
        self.attn = MultiHeadAttentionEuclid(
            d_model=d_model,
            num_heads=n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            use_rope=use_rope,
            rope_max_seq_len=rope_max_seq_len,
        )
        self.ln_ffn = LayerNorm(d_model, eps=ln_eps)
        self.ffn = FeedForward(d_model=d_model, d_hidden=d_ff, dropout=dropout)
        self.drop_resid = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln_attn(x)
        z = self.attn(h, mask=mask)          # (B,T,D)
        if isinstance(z, tuple):              # if someone passes return_attn=True by mistake
            z, _ = z
        x = x + self.drop_resid(z)
        y = self.ffn(self.ln_ffn(x))
        x = x + self.drop_resid(y)
        return x


class EuclideanEncoder(nn.Module):
    """
    Minimal encoder-only stack with token embeddings.
    Matches your train.py usage:
        EuclideanEncoder(vocab_size, d_model, n_heads, d_ff, n_layers, dropout)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_rope: bool = False,
        rope_max_seq_len: int = 4096,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_rope=use_rope,
                rope_max_seq_len=rope_max_seq_len,
                ln_eps=ln_eps,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = LayerNorm(d_model, eps=ln_eps)

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B,T)
        attention_mask: Optional[torch.Tensor] = None,  # (B,T) 1=real, 0=pad
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)          # (B,T,D)
        x = self.dropout(x)
        mask = make_padding_mask(attention_mask) if attention_mask is not None else None  # (B,1,T,T) bool
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_final(x)
        return x