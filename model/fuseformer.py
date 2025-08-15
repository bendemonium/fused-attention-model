# model/fuseformer.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .more_model_utils import LayerNorm, FeedForward, make_padding_mask
from .euclidean import MultiHeadAttentionEuclid as EuclidMHA
from .hyperbolic import LorentzSelfAttention              # returns points when return_points=True
from .fusion import FusionOnManifold                      # on-manifold barycentric fusion + log-map->D
from .fuseformerconfig import FuseFormerConfig           # NOTE: underscore in filename

class FuseFormerBlock(nn.Module):
    """
    Pre-LN mixed-geometry block:
      x -> LN -> [Euclid MHA (B,T,D), Hyperbolic MHA (B,T,A)] -> on-manifold fusion -> +res
        -> LN -> FFN (Euclidean) -> +res
    Notes:
      - Fusion module holds a per-block Lorentz anchor as a geoopt.ManifoldParameter.
      - A = lorentz_spatial_dim + 1 (ambient).
    """
    def __init__(self, cfg: FuseFormerConfig):
        super().__init__()
        D = cfg.d_model
        H = cfg.num_heads
        n = cfg.lorentz_spatial_dim
        A = n + 1

        self.ln_attn = LayerNorm(D, eps=cfg.layer_norm_eps)

        # Euclidean self-attention (returns tensor or (tensor, attn))
        self.euclid_attn = EuclidMHA(
            d_model=D,
            num_heads=H,
            attn_dropout=cfg.attn_dropout,
        )

        # Hyperbolic self-attention (can return manifold points for fusion)
        self.hyp_attn = LorentzSelfAttention(
            d_model=D,
            num_heads=H,
            spatial_dim=n,
            tau=cfg.lorentz_tau,
            attn_dropout=cfg.attn_dropout,
            rope_max_seq_len=cfg.rope_max_seq_len,
            karcher_steps=cfg.karcher_steps,
            pre_lift_scale=getattr(cfg, "pre_lift_scale", 1.0),
        )

        # On-manifold fusion (owns a learned Lorentz anchor)
        self.fuse = FusionOnManifold(
            d_model=D,
            ambient_dim=A,
            gate_hidden=cfg.fusion_gate_hidden,
            fusion_karcher_steps=cfg.fusion_karcher_steps,
        )

        self.ln_ffn = LayerNorm(D, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(D, cfg.d_ff, dropout=cfg.dropout, activation="gelu")
        self.drop_resid = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                    # (B,T,D)
        attn_mask: Optional[torch.Tensor],  # (B,1,T,T) True=keep
        return_alpha: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.ln_attn(x)

        # Euclidean branch
        euclid_out = self.euclid_attn(h, mask=attn_mask)
        if isinstance(euclid_out, tuple):
            e_out, _ = euclid_out      # we don't use Euclid attn weights here
        else:
            e_out = euclid_out         # (B,T,D)

        # Hyperbolic branch: get (B,T,A) manifold points, sharing fusion anchor
        y_h = self.hyp_attn(
            h,
            anchor_p=self.fuse.anchor,    # keep tangent/anchor consistent with fusion
            mask=attn_mask,
            return_points=True
        )  # (B,T,A)

        # On-manifold fusion → Euclidean D
        z_fused, alpha = self.fuse(e_out, y_h)  # (B,T,D), (B,T,1)
        x = x + self.drop_resid(z_fused)

        # Euclidean FFN
        z = self.ffn(self.ln_ffn(x))
        x = x + self.drop_resid(z)
        return (x, alpha if return_alpha else None)


class FuseFormerEncoder(nn.Module):
    """
    Token embedding -> L × FuseFormerBlock -> final LayerNorm.
    """
    def __init__(self, cfg: FuseFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([FuseFormerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_final = LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,                       # (B,T)
        attention_mask: Optional[torch.Tensor] = None, # (B,T) 1=real, 0=pad
        return_alphas: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.embed_tokens(input_ids)          # (B,T,D)
        x = self.dropout(x)
        mask = make_padding_mask(attention_mask) if attention_mask is not None else None

        alpha_logs: List[torch.Tensor] = []
        for blk in self.layers:
            x, alpha = blk(x, mask, return_alpha=return_alphas)
            if return_alphas and alpha is not None:
                alpha_logs.append(alpha)          # each (B,T,1)

        x = self.ln_final(x)
        return x, (alpha_logs if return_alphas else None)


class FuseFormerForMaskedLM(PreTrainedModel):
    config_class = FuseFormerConfig

    def __init__(self, config: FuseFormerConfig):
        super().__init__(config)
        self.encoder = FuseFormerEncoder(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tie_weights()

    # --- Embedding tying APIs expected by HF ---

    def get_input_embeddings(self) -> nn.Embedding:
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.encoder.embed_tokens = value
        if getattr(self.config, "tie_word_embeddings", True) and hasattr(self, "lm_head"):
            # weight tying
            self.lm_head.weight = self.encoder.embed_tokens.weight

    def get_output_embeddings(self):
        return None if getattr(self.config, "tie_word_embeddings", True) else self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            if hasattr(self, "lm_head"):
                self.lm_head.weight = self.encoder.embed_tokens.weight

    # --- Forward ---

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,   # -100 to ignore
        output_alphas: bool = False,
        **unused: Any,
    ) -> Dict[str, torch.Tensor]:
        hidden, alphas = self.encoder(input_ids, attention_mask, return_alphas=output_alphas)

        if getattr(self.config, "tie_word_embeddings", True):
            logits = torch.matmul(hidden, self.encoder.embed_tokens.weight.t())  # (B,T,V)
        else:
            logits = self.lm_head(hidden)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if output_alphas and alphas:
            out["alphas"] = torch.stack(alphas, dim=0)  # (L,B,T,1)

        if labels is not None:
            vocab = logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            out["loss"] = loss_fct(logits.view(-1, vocab), labels.view(-1))

        return out