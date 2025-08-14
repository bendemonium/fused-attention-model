# models/more_model_utils.py
# Shared building blocks used by euclidean/hyperbolic/fuseformer modules.
# - FP32 LayerNorm (stable)
# - FeedForward MLP (Euclidean)
# - Rotary Positional Embeddings (RoPE)
# - Padding mask helper

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Masks
# ------------------------------

def make_padding_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    attention_mask: (B, T) with 1 for real tokens, 0 for pad.
    Returns (B, 1, T, T) broadcastable mask for attention.
    """
    if attention_mask is None:
        return None
    m = (attention_mask > 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
    return m & m.transpose(-1, -2)                      # (B,1,T,T)


# ------------------------------
# LayerNorm (fp32 compute)
# ------------------------------

class LayerNorm(nn.Module):
    """FP32 LayerNorm for stability (keeps params in model dtype)."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(-1, keepdim=True)
        var = x32.var(-1, unbiased=False, keepdim=True)
        y = (x32 - mean) / (var + self.eps).sqrt()
        y = y.to(orig_dtype)
        return y * self.weight + self.bias


# ------------------------------
# Feed-Forward MLP (Euclidean)
# ------------------------------

class FeedForward(nn.Module):
    """
    Simple 2-layer MLP with activation and dropout.
    Use after fusion/log-map (shared across all variants).
    """
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        y = self.fc2(h)
        y = self.dropout(y)
        return y


# ------------------------------
# Rotary Positional Embeddings (RoPE)
# ------------------------------

class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin for RoPE. Works on per-head dims; requires even head_dim.
    Call with q,k shaped (B,H,T,Dh). Returns rotated (q,k).
    """
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
        """
        q,k: (B, H, T, Dh)
        returns q_rot, k_rot with RoPE applied along the last dim.
        """
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