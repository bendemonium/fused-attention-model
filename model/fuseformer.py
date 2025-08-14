from typing import Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict

# --- helpers (math utilities you already have) ---
from euclidean import (
    split_heads, combine_heads, scaled_dot_product_attention as e_scaled_attn,
    build_rope_cache, apply_rope, layer_norm, linear, ffn as euclid_ffn,
    make_padding_mask, make_causal_mask, merge_masks,
)
from hyperbolic import (
    origin, log_map, exp_map, pairwise_scores_lorentz, safe_softmax,
    frechet_mean, pack_heads, append_time_and_lift, combine_heads_ambient,
)
from model.fusion import fuse_default, summarize_alpha

from fuseformerconfig import FuseformerConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel

# =========================
# Modules
# =========================
class EuclidMHA(nn.Module):
    dim: int
    num_heads: int
    dropout: float = 0.0
    use_rope: bool = True  # apply RoPE on Q,K if True

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True, rope_cos=None, rope_sin=None):
        q = nn.Dense(self.dim, name="q_proj")(x)
        k = nn.Dense(self.dim, name="k_proj")(x)
        v = nn.Dense(self.dim, name="v_proj")(x)

        qh, kh, vh = split_heads(q, self.num_heads), split_heads(k, self.num_heads), split_heads(v, self.num_heads)

        if self.use_rope:
            assert rope_cos is not None and rope_sin is not None, "RoPE caches required"
            qh, kh = apply_rope(qh, kh, rope_cos, rope_sin)

        ctx, attn = e_scaled_attn(qh, kh, vh, mask=mask)
        z = combine_heads(ctx)
        z = nn.Dense(self.dim, name="o_proj")(z)
        z = nn.Dropout(self.dropout)(z, deterministic=deterministic)
        return z, attn


class LorentzMHA(nn.Module):
    dim_model: int        # output projection to model dim
    num_heads: int
    spatial_dim: int      # n; ambient is n+1 per head
    tau: float = 1.0
    dropout: float = 0.0
    karcher_steps: int = 1
    use_rope: bool = True

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True, rope_cos=None, rope_sin=None):
        # Project to per-head spatial vectors (pack heads): (B,T,H*n)
        H, n = self.num_heads, self.spatial_dim
        u_q = nn.Dense(H * n, name="q_lift")(x)
        u_k = nn.Dense(H * n, name="k_lift")(x)
        u_v = nn.Dense(H * n, name="v_lift")(x)

        # Split heads → (B,H,T,n)
        uq, uk, uv = pack_heads(u_q, H), pack_heads(u_k, H), pack_heads(u_v, H)

        # Optional RoPE on these spatial vectors before lift
        if self.use_rope:
            assert rope_cos is not None and rope_sin is not None, "RoPE caches required"
            uq, uk = apply_rope(uq, uk, rope_cos, rope_sin)

        # Append time coord (0) and exp at origin → ℍ^n ambient (n+1)
        qh = append_time_and_lift(uq)  # (B,H,T,n+1)
        kh = append_time_and_lift(uk)
        vh = append_time_and_lift(uv)

        # Scores = -tau * d_H^2; masked softmax
        scores = pairwise_scores_lorentz(qh, kh, tau=self.tau)  # (B,H,T,T)
        a = safe_softmax(scores, mask=mask, axis=-1)            # (B,H,T,T)

        # Weighted Fréchet mean around query points (Karcher steps)
        v_exp = vh[:, :, None, :, :]                            # (B,H,1,T,D)
        v_exp = jnp.broadcast_to(v_exp, (*a.shape, vh.shape[-1]))
        y = frechet_mean(v_exp, a, anchor=qh, steps=self.karcher_steps)  # (B,H,T,n+1)

        # Log at origin → combine heads → Dense to model dim
        o = origin(n, dtype=y.dtype)
        o_bht = jnp.broadcast_to(o, y.shape)
        y_log = log_map(o_bht, y)                               # (B,H,T,n+1)
        z = y_log.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1)
        z = nn.Dense(self.dim_model, name="o_proj")(z)
        z = nn.Dropout(self.dropout)(z, deterministic=deterministic)
        return z, a


class FusionModule(nn.Module):
    dim: int
    gate_hidden: int = 64

    @nn.compact
    def __call__(self, euclid_vec, lorentz_point, deterministic=True):
        """
        euclid_vec:   (B,T,D) Euclid path (after MHA)
        lorentz_point:(B,T,D_l= n+1 per head packed and projected) -> here already projected to D via LorentzMHA
        We do tangent-space fusion assuming LorentzMHA returned a Euclidean vector (log+linear).
        """
        # In this design LorentzMHA already returned Euclidean z of shape (B,T,D).
        # If you want to fuse true ℍ^n points, swap to fuse_default(...) with return_space="tangent".
        e = nn.Dense(self.dim, name="U_e")(euclid_vec)
        h = nn.Dense(self.dim, name="U_h")(lorentz_point)

        feats = jnp.concatenate([e, h], axis=-1)
        g = nn.Dense(self.gate_hidden, name="gate_h")(feats)
        g = nn.tanh(g)
        alpha = nn.Dense(1, name="gate_out")(g)
        alpha = nn.sigmoid(alpha)  # (B,T,1)

        u = alpha * h + (1.0 - alpha) * e
        return u, jnp.squeeze(alpha, axis=-1)


class EncoderBlock(nn.Module):
    cfg: FuseformerConfig

    @nn.compact
    def __call__(self, x, self_mask=None, deterministic=True, rope_cos=None, rope_sin=None):
        D = self.cfg.hidden_size
        H_e = self.cfg.head_split["euclid"]
        H_h = self.cfg.head_split["lorentz"]
        use_rope = (self.cfg.position_encoding_type == "rope")

        # Pre-LN
        h = nn.LayerNorm(self.cfg.layer_norm_eps, name="pre_ln_attn")(x)

        # Euclid attention
        e_out, _ = EuclidMHA(D, H_e, dropout=self.cfg.attention_dropout, use_rope=use_rope, name="euclid_attn")(
            h, mask=self_mask, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin
        )
        # Lorentz attention (returns Euclidean-projected vector)
        h_out, _ = LorentzMHA(D, H_h, self.cfg.lorentz_spatial_dim, tau=self.cfg.lorentz_tau,
                              dropout=self.cfg.attention_dropout, karcher_steps=self.cfg.karcher_steps,
                              use_rope=use_rope, name="lorentz_attn")(
            h, mask=self_mask, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin
        )

        fused, alpha = FusionModule(D, gate_hidden=self.cfg.fusion_gate_hidden, name="fusion")(e_out, h_out, deterministic=deterministic)
        x = x + fused

        # FFN
        h2 = nn.LayerNorm(self.cfg.layer_norm_eps, name="pre_ln_ffn")(x)
        h2 = nn.Dense(self.cfg.ffn_hidden_size, name="ffn_in")(h2)
        h2 = nn.gelu(h2)
        h2 = nn.Dropout(self.cfg.dropout)(h2, deterministic=deterministic)
        h2 = nn.Dense(D, name="ffn_out")(h2)
        h2 = nn.Dropout(self.cfg.dropout)(h2, deterministic=deterministic)
        return x + h2


class DecoderBlock(nn.Module):
    cfg: FuseformerConfig

    @nn.compact
    def __call__(self, y, enc_out, self_mask=None, cross_mask=None, deterministic=True, rope_cos=None, rope_sin=None):
        D = self.cfg.hidden_size
        H_e = self.cfg.head_split["euclid"]
        H_h = self.cfg.head_split["lorentz"]
        use_rope = (self.cfg.position_encoding_type == "rope")

        # Masked self-attn
        hs = nn.LayerNorm(self.cfg.layer_norm_eps, name="pre_ln_self")(y)
        e_self, _ = EuclidMHA(D, H_e, dropout=self.cfg.attention_dropout, use_rope=use_rope, name="euclid_self")(
            hs, mask=self_mask, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin
        )
        h_self, _ = LorentzMHA(D, H_h, self.cfg.lorentz_spatial_dim, tau=self.cfg.lorentz_tau,
                               dropout=self.cfg.attention_dropout, karcher_steps=self.cfg.karcher_steps,
                               use_rope=use_rope, name="lorentz_self")(
            hs, mask=self_mask, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin
        )
        fused_self, _ = FusionModule(D, gate_hidden=self.cfg.fusion_gate_hidden, name="fusion_self")(e_self, h_self, deterministic=deterministic)
        y = y + fused_self

        # Cross-attn: Euclid and Lorentz paths over (K,V)=enc_out
        hc = nn.LayerNorm(self.cfg.layer_norm_eps, name="pre_ln_cross")(y)

        # Euclid cross
        q = nn.Dense(D, name="e_q_proj")(hc)
        k = nn.Dense(D, name="e_k_proj")(enc_out)
        v = nn.Dense(D, name="e_v_proj")(enc_out)
        qe, ke, ve = split_heads(q, H_e), split_heads(k, H_e), split_heads(v, H_e)
        if use_rope:
            Tq = qe.shape[2]
            Dh = qe.shape[-1]
            rope_cos_cross, rope_sin_cross = build_rope_cache(Tq, Dh, dtype=qe.dtype)
            qe, ke = apply_rope(qe, ke, rope_cos_cross, rope_sin_cross)
        ctx_e, _ = e_scaled_attn(qe, ke, ve, mask=cross_mask)
        ctx_e = combine_heads(ctx_e)
        ctx_e = nn.Dense(D, name="e_o_proj")(ctx_e)

        # Lorentz cross (lift enc_out too)
        Hh, n = H_h, self.cfg.lorentz_spatial_dim
        uq = nn.Dense(Hh * n, name="h_q_lift")(hc)
        uk = nn.Dense(Hh * n, name="h_k_lift")(enc_out)
        uv = nn.Dense(Hh * n, name="h_v_lift")(enc_out)
        uq, uk, uv = pack_heads(uq, Hh), pack_heads(uk, Hh), pack_heads(uv, Hh)
        if use_rope:
            Tq = uq.shape[2]
            Dh = uq.shape[-1]
            rope_cos_cross, rope_sin_cross = build_rope_cache(Tq, Dh, dtype=uq.dtype)
            uq, uk = apply_rope(uq, uk, rope_cos_cross, rope_sin_cross)
        qh = append_time_and_lift(uq)
        kh = append_time_and_lift(uk)
        vh = append_time_and_lift(uv)
        scores = pairwise_scores_lorentz(qh, kh, tau=self.cfg.lorentz_tau)
        a = safe_softmax(scores, mask=cross_mask, axis=-1)
        v_exp = vh[:, :, None, :, :]
        v_exp = jnp.broadcast_to(v_exp, (*a.shape, vh.shape[-1]))
        y_man = frechet_mean(v_exp, a, anchor=qh, steps=self.cfg.karcher_steps)
        o = origin(n, dtype=y_man.dtype)
        o_bht = jnp.broadcast_to(o, y_man.shape)
        y_log = log_map(o_bht, y_man)
        ctx_h = y_log.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[1], -1)
        ctx_h = nn.Dense(D, name="h_o_proj")(ctx_h)

        fused_cross, _ = FusionModule(D, gate_hidden=self.cfg.fusion_gate_hidden, name="fusion_cross")(ctx_e, ctx_h, deterministic=deterministic)
        y = y + fused_cross

        # FFN
        hf = nn.LayerNorm(self.cfg.layer_norm_eps, name="pre_ln_ffn")(y)
        hf = nn.Dense(self.cfg.ffn_hidden_size, name="ffn_in")(hf)
        hf = nn.gelu(hf)
        hf = nn.Dropout(self.cfg.dropout)(hf, deterministic=deterministic)
        hf = nn.Dense(D, name="ffn_out")(hf)
        hf = nn.Dropout(self.cfg.dropout)(hf, deterministic=deterministic)
        y = y + hf
        return y


class FuseformerModel(nn.Module):
    cfg: FuseformerConfig

    def setup(self):
        self.token_embed = nn.Embed(self.cfg.vocab_size, self.cfg.hidden_size, name="token_embedding")
        # position: we’re using RoPE, so no learned position embed by default
        self.encoder_layers = [EncoderBlock(self.cfg, name=f"encoder_{i}") for i in range(self.cfg.num_encoder_layers)]
        self.decoder_layers = [DecoderBlock(self.cfg, name=f"decoder_{i}") for i in range(self.cfg.num_decoder_layers)]
        if not self.cfg.tie_word_embeddings:
            self.lm_head = nn.Dense(self.cfg.vocab_size, name="lm_head")

    def _rope_caches(self, T: int, Dh: int, dtype):
        # Dh must be even
        if Dh % 2 == 1:
            Dh = Dh - 1
        return build_rope_cache(T, Dh, dtype=dtype)

    def encode(self, input_ids, attention_mask=None, deterministic=True):
        x = self.token_embed(input_ids)
        # Build self-attn padding mask
        sa_mask = None
        if attention_mask is not None:
            sa_mask = make_padding_mask(attention_mask)
        # RoPE caches for per-head dims
        if self.cfg.position_encoding_type == "rope":
            H_e = self.cfg.head_split["euclid"]
            H_h = self.cfg.head_split["lorentz"]
            Dh_e = self.cfg.hidden_size // H_e
            Dh_h = self.cfg.lorentz_spatial_dim
            Dh = min(Dh_e, Dh_h)
            rope_cos, rope_sin = self._rope_caches(x.shape[1], Dh, x.dtype)
        else:
            rope_cos = rope_sin = None

        for layer in self.encoder_layers:
            x = layer(x, self_mask=sa_mask, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin)
        return x

    def decode(self, decoder_input_ids, enc_out, decoder_attention_mask=None, encoder_attention_mask=None, deterministic=True):
        y = self.token_embed(decoder_input_ids)

        # Causal self mask
        T = decoder_input_ids.shape[1]
        causal = make_causal_mask(T)  # (1,1,T,T)
        self_mask = causal
        if decoder_attention_mask is not None:
            pad = make_padding_mask(decoder_attention_mask)
            self_mask = merge_masks(self_mask, pad)

        # Cross mask: decoder tokens (rows) can only attend to non-pad encoder tokens (cols)
        cross_mask = None
        if (encoder_attention_mask is not None) and (decoder_attention_mask is not None):
            cross_mask = decoder_attention_mask[:, None, :, None] & encoder_attention_mask[:, None, None, :]

        if self.cfg.position_encoding_type == "rope":
            H_e = self.cfg.head_split["euclid"]
            H_h = self.cfg.head_split["lorentz"]
            Dh_e = self.cfg.hidden_size // H_e
            Dh_h = self.cfg.lorentz_spatial_dim
            Dh = min(Dh_e, Dh_h)
            rope_cos, rope_sin = self._rope_caches(y.shape[1], Dh, y.dtype)
        else:
            rope_cos = rope_sin = None

        for layer in self.decoder_layers:
            y = layer(y, enc_out, self_mask=self_mask, cross_mask=cross_mask, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin)
        return y

    def __call__(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        labels: Optional[jnp.ndarray] = None,
        deterministic=True,
    ):
        enc_out = self.encode(input_ids, attention_mask, deterministic=deterministic)
        dec_out = self.decode(decoder_input_ids, enc_out, decoder_attention_mask, attention_mask, deterministic=deterministic)

        if self.cfg.tie_word_embeddings:
            logits = dec_out @ self.variables["params"]["token_embedding"]["embedding"].T
        else:
            logits = self.lm_head(dec_out)

        out = {"logits": logits}
        if labels is not None:
            vocab = logits.shape[-1]
            mask = (labels != -100)
            onehot = jnp.where(mask[..., None], jnp.eye(vocab, dtype=logits.dtype)[labels], 0.0)
            logp = nn.log_softmax(logits, axis=-1)
            nll = -jnp.sum(onehot * logp, axis=-1)
            loss = (nll * mask).sum() / jnp.maximum(mask.sum(), 1)
            out["loss"] = loss
        return out


# =========================
# HF wrapper (Flax)
# =========================
class FlaxFuseformerForConditionalGeneration(FlaxPreTrainedModel):
    config_class = FuseformerConfig
    module_class = FuseformerModel

    def __init__(self, config: FuseformerConfig, dtype=jnp.float32, **kwargs):
        module = self.module_class(config)
        super().__init__(config, module, dtype=dtype, **kwargs)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape=(1, 8), decoder_input_shape=(1, 8)) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype=jnp.int32)
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype=jnp.int32)
        variables = self.module.init(
            rng,
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=jnp.ones_like(input_ids),
            decoder_attention_mask=jnp.ones_like(decoder_input_ids),
            labels=None,
            deterministic=True,
        )
        return variables["params"]

    def __call__(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        labels: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        params: Optional[FrozenDict] = None,
        **kwargs: Any,
    ):
        return self.module.apply(
            {"params": params or self.params},
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            deterministic=deterministic,
        )