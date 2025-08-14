# models/fuseformer_config.py
from __future__ import annotations
from typing import Optional
from transformers import PretrainedConfig


class FuseFormerConfig(PretrainedConfig):
    model_type = "fuseformer"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        rope_max_seq_len: int = 4096,

        # Hyperbolic (Lorentz) branch
        lorentz_spatial_dim: Optional[int] = None,  # per-head n; ambient is n+1
        lorentz_tau: float = 1.0,
        karcher_steps: int = 1,

        # Fusion
        fusion_gate_hidden: int = 64,
        fusion_karcher_steps: int = 1,

        # Misc
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,

        # Standard HF extras
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Core dims
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.d_ff = int(d_ff)

        # Dropouts / pos enc
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.rope_max_seq_len = int(rope_max_seq_len)

        # Hyperbolic specifics
        self.lorentz_spatial_dim = int(lorentz_spatial_dim) if lorentz_spatial_dim is not None else (self.d_model // self.num_heads)
        self.lorentz_tau = float(lorentz_tau)
        self.karcher_steps = int(karcher_steps)

        # Fusion
        self.fusion_gate_hidden = int(fusion_gate_hidden)
        self.fusion_karcher_steps = int(fusion_karcher_steps)

        # Misc
        self.layer_norm_eps = float(layer_norm_eps)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.initializer_range = float(initializer_range)

        # -------- Validation / derived checks --------
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}).")

        per_head_dim = self.d_model // self.num_heads
        # RoPE requires even per-head dims (both for Euclid QK and Lorentz pre-lift)
        if per_head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even per-head dim; got d_model/num_heads = {per_head_dim}.")

        if self.lorentz_spatial_dim % 2 != 0:
            raise ValueError(f"RoPE on Lorentz pre-lift requires even lorentz_spatial_dim; got {self.lorentz_spatial_dim}.")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0.")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")