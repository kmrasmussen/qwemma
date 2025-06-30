import dataclasses
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax import linen
import enum

LayerCache = dict[str, jax.Array]

K_MASK = -2.3819763e38  # Set to a large negative number.

import os
os.environ["JAX_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"  
jax.config.update("jax_default_matmul_precision", "bfloat16")  # or "float32"
jax.config.update("jax_enable_x64", True)

class Einsum(nn.Module):
  shape: tuple[int, ...]
  weight_name: str = 'w'
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  dtype: jnp.dtype | None = None

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(
        self.weight_name,
        self.initializer,
        self.shape,
        self.dtype if self.dtype is not None else None,
    )
    ret = jnp.einsum(eqn, x, w)
    return ret

class QwemmaRMSNorm(linen.Module):
  @linen.compact
  def __call__(self, x):
    self.sow('intermediates', 'input', x)
    scale = self.param('scale', linen.initializers.zeros_init(), (x.shape[-1]))
    x_float32 = x.astype(jnp.float32) #jnp.asarray(x, jnp.float32) # 
    self.sow('intermediates', 'x_float32', x_float32)
    x_float32_squared = jnp.square(x_float32)
    self.sow('intermediates', 'x_float32_squared', x_float32_squared)
    variance = jnp.mean(
      jnp.square(x_float32),
      axis=-1,
      keepdims=True,
      dtype=jnp.float32
    )
    self.sow('intermediates', 'variacne', variance)
    rescaled_x = x_float32 * jax.lax.rsqrt(variance + 1e-6) #* jax.lax.rsqrt(variance + 1e-6)
    self.sow('intermediates', 'rescaled_x', rescaled_x)
    rescaled_x_bfloat16 = rescaled_x.astype(jnp.bfloat16)
    self.sow('intermediates', 'rescaled_x_bfloat16', rescaled_x_bfloat16)
    normed_inputs = rescaled_x_bfloat16 * scale
    self.sow('intermediates', 'normed_inputs', normed_inputs)
    return normed_inputs

class QwemmaEmbedder(nn.Module):
  vocab_size: int
  embed_dim: int
  vision_proj_dim: int | None = None
  qwemma_setting_rescale_embeddings: bool = True

  def setup(self):
    self.input_embedding_table = self.param(
        'input_embedding',
        nn.initializers.normal(),
        (self.vocab_size, self.embed_dim),
    )

  def encode(self, x: jax.Array) -> jax.Array:
    x = self.input_embedding_table[(x,)]
    self.sow('intermediates', 'inspect.embedder.x', x)
    return x

  def decode(self, x: jax.Array) -> jax.Array:
    """Decodes the input vectors.

    Args:
      x: Array of shape [seq_len, embed_dim] or [batch_size, seq_len,
        embed_dim].

    Returns:
      Array of shape [seq_len, vocab_size] or [batch_size, seq_len, vocab_size].
    """
    return jnp.dot(x, self.input_embedding_table.T)

class QwemmaAttention(nn.Module):
  num_heads: int
  num_kv_heads: int
  features: int
  head_dim: int
  query_pre_attn_scalar: float
  rope_base_frequency: int

  def setup(self):
    self.attn_vec_einsum = Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

    self.q_einsum = Einsum(
        shape=(self.num_heads, self.features, self.head_dim),
    )
    self.kv_einsum = Einsum(
        shape=(2, self.num_kv_heads, self.features, self.head_dim),
    )
    self._query_norm = QwemmaRMSNorm()
    self._key_norm = QwemmaRMSNorm()

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    self.sow('intermediates', 'inspect.input', x)
    query_proj_prenorm = self.q_einsum('BTD,NDH->BTNH', x)
    self.sow('intermediates', 'inspect.q_proj_prenorm', query_proj_prenorm)
    key_proj_prenorm, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)
    self.sow('intermediates', 'inspect.key_proj_prenorm', key_proj_prenorm)
    self.sow('intermediates', 'inspect.value_proj', value_proj)

    query_proj_preroped = self._query_norm(query_proj_prenorm)
    self.sow('intermediates', 'inspect.query_norm.out', query_proj_preroped)

    key_proj_preroped = self._key_norm(key_proj_prenorm)
    self.sow('intermediates', 'inspect.key_norm.out', key_proj_preroped)

    query_proj = self.apply_rope(
        query_proj_preroped,
        segment_pos,
        base_frequency=self.rope_base_frequency,
    )
    self.sow('intermediates', 'inspect.query_roped.out', query_proj)

    key_proj = self.apply_rope(
        key_proj_preroped,
        segment_pos,
        base_frequency=self.rope_base_frequency,
    )
    self.sow('intermediates', 'inspect.key_roped.out', key_proj)

    # Cache is left aligned.
    # Save the KV values to the cache.
    if cache is not None:
      end_index = cache['end_index'][0]
      cache_size = cache['v'].shape[1]
      slice_indices = (0, end_index % cache_size, 0, 0)

      # [batch_size, cache_size, num_heads, head_dim]
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )

      # [batch_size, cache_size, num_heads, head_dim]
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    # BEGIN Actual self-attention part

    # Reshape matrices to enable einsums over groups.
    b, t, kg, h = query_proj.shape
    query_proj = query_proj.reshape(
        (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
    )
    logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_proj, key_proj)
    self.sow('intermediates', 'inspect.actual_attention.logits_einsum', logits)
    logits_float32 = logits.astype(jnp.float32)
    logits_scaled_float32 = logits_float32 * self.query_pre_attn_scalar
    logits = logits_scaled_float32.astype(jnp.bfloat16)
    self.sow('intermediates', 'inspect.actual_attention.query_pre_attention_scalar', self.query_pre_attn_scalar)
    self.sow('intermediates', 'inspect.actual_attention.logits_scaled', logits)

    b, t, k, g, s = logits.shape
    logits = logits.reshape((b, t, k * g, s))

    # [batch_size, seq_len, num_heads, cache_size]
    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    self.sow('intermediates', 'inspect.actual_attention.attn_mask', attn_mask)
    self.sow('intermediates', 'inspect.actual_attention.padded_logits', padded_logits)

    # Multi-head attention matrices.
    # [batch_size, seq_len, num_heads, cache_size]
    probs = jax.nn.softmax(padded_logits.astype(jnp.float32), axis=-1).astype(key_proj.dtype)
    self.sow('intermediates', 'inspect.actual_attention.probs', probs)

    # Reshape matrices to enable einsums over groups.
    b, t, kg, h = probs.shape
    probs = probs.reshape(
        (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
    )
    encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
    b, t, k, g, h = encoded.shape
    encoded = encoded.reshape((b, t, k * g, h))

    # END Actual self-attention part

    self.sow('intermediates', 'inspect.attn_encoded', encoded)
    # [batch_size, seq_len, features]
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)
    self.sow('intermediates', 'inspect.o_proj.out', attn_output)

    if cache is not None:
      seq_len = x.shape[1]
      new_cache = {
          # [batch_size, cache_size, num_heads, head_dim]
          'v': value_proj,
          # [batch_size, cache_size, num_heads, head_dim]
          'k': key_proj,
          # [batch_size]
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @classmethod
  def init_cache(
      cls,
      cache_size: int,
      num_heads: int,
      head_dim: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> LayerCache:
    del cls  # not used
    return {
        'v': jnp.zeros(
            (batch_size, cache_size, num_heads, head_dim), dtype=dtype
        ),
        'k': jnp.zeros(
            (batch_size, cache_size, num_heads, head_dim), dtype=dtype
        ),
        'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
    }

  @classmethod
  def apply_rope(
      self,
      inputs: jax.Array,
      positions: jax.Array,
      *,
      base_frequency: int,
  ) -> jax.Array:
    head_dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base_frequency**fraction

    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    # trying to match hf dtypes:
    sin = sin.astype(inputs.dtype)
    cos = cos.astype(inputs.dtype)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    out = out.astype(inputs.dtype)
    return out

class QwemmaFeedForward(nn.Module):
  """Feed forward module."""

  features: int  # features = embed_dim
  hidden_dim: int
  transpose_gating_einsum: bool

  @nn.compact
  def __call__(self, x):
    # Some versions use an alternate parameter ordering that
    # transposes hidden_dim and features.
    if self.transpose_gating_einsum:
      eq = '...F,NHF->...NH'
      gating = Einsum(
          shape=(2, self.hidden_dim, self.features),
          weight_name='gating_einsum',
      )
    else:
      eq = '...F,NFH->...NH'
      gating = Einsum(
          shape=(2, self.features, self.hidden_dim),
          weight_name='gating_einsum',
      )

    # Use the same scope for backwards compatibility with existing checkpoints
    # created before using `layers.Einsum` here.
    nn.share_scope(self, gating)

    # [batch_size, seq_len, 2, hidden_dim]
    gate = gating(eq, x)
    # [batch_size, seq_len, hidden_dim]

    # qwen uses silu instead of gelu
    #activations = nn.gelu(gate[..., 0, :]) * gate[..., 1, :]
    gate_proj_out = gate[..., 0, :]
    self.sow('intermediates', 'inspect.mlp.gate_proj_out', gate_proj_out)
    up_proj_out = gate[..., 1, :]
    self.sow('intermediates', 'inspect.mlp.up_proj_out', up_proj_out)
    gate_activations = nn.silu(gate_proj_out)
    self.sow('intermediates', 'inspect.mlp.act_fn_out', gate_activatoins)
    activations = gate_activations * up_proj_out
    #activations = nn.silu(gate[..., 0, :]) * gate[..., 1, :]
    self.sow('intermediates', 'inspect.mlp.product_out', activations)

    # Project back from hidden_dim to features.
    linear = Einsum(
        shape=(self.hidden_dim, self.features),
        weight_name='linear',
    )
    nn.share_scope(self, linear)

    # [batch_size, seq_len, features]
    outputs = linear('...H,HF->...F', activations)
    self.sow('intermediates', 'inspect.mlp.down_proj_out', outputs)

    return outputs

class QwemmaBlock(nn.Module):
  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int
  query_pre_attn_scalar: float
  transpose_gating_einsum: bool
  rope_base_frequency: int

  def setup(self):
    self.pre_attention_norm = QwemmaRMSNorm()
    self.attn = QwemmaAttention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        rope_base_frequency=self.rope_base_frequency,
    )
    self.post_attention_norm = QwemmaRMSNorm()
    self.mlp = QwemmaFeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
    )

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array]:
    self.sow('intermediates', 'inspect.input', x)
    inputs_normalized = self.pre_attention_norm(x)
    self.sow('intermediates', 'inspect.pre_attention_norm_out', inputs_normalized)

    # attn_output.shape = [batch_size, seq_len, embed_dim]
    # cache["k"].shape = [batch_size, cache_size, num_heads, head_dim]
    # cache["v"].shape = [batch_size, cache_size, num_heads, head_dim]
    # cache["end_index"].shape = [batch_size]
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    self.sow('intermediates', 'inspect.attn_out', attn_output)
    attn_output += x
    self.sow('intermediates', 'inspect.attn_added_out', attn_output)
    residual = attn_output
    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)
      self.sow('intermediates', 'inspect.post_attention_norm_out', attn_output)
    outputs = self.mlp(attn_output)
    self.sow('intermediates', 'inspect.mlp_out', outputs)
    outputs += residual
    self.sow('intermediates', 'inspect.block_out', outputs)
    return cache, outputs
