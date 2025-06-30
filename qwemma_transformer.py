from __future__ import annotations

import math
import dataclasses
import functools
from typing import Any, ClassVar
import jax
import einops
import flax
from flax import linen as nn
#from gemma import transformer
from gemma.gm.utils import _attention_mask
#from gemma.gm.utils import _dtype_params
#from gemma.gm.utils import _jax_utils
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.typing import Bool, Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member

_PADDING_ID = 0

import modeling_qwemma as modules


import os
os.environ["JAX_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"  
jax.config.update("jax_default_matmul_precision", "bfloat16")  # or "float32"
jax.config.update("jax_enable_x64", True)


Cache = dict[str, modules.LayerCache]

def build_positions_from_mask(input_mask: jax.Array) -> jax.Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)

@dataclasses.dataclass(frozen=True)
class QwemmaTransformerConfig:
  """Configuration for the gemma transformer."""
  num_layers: int
  num_embed: int  # TODO(epot): Rename to `vocab_size` for consistency.
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  global_base_frequency: int 
  transpose_gating_einsum: bool = False
  query_pre_attn_scalar: float | None = None

  @classmethod
  def qwemma_06b(cls, qwen_model_config):
    return cls(
        num_layers=2, #"qwen_model_config['num_hidden_layers'],
        num_embed=qwen_model_config['vocab_size'],
        embed_dim=qwen_model_config['hidden_size'],
        hidden_dim=qwen_model_config['intermediate_size'],
        num_heads=qwen_model_config['num_attention_heads'],
        head_dim=qwen_model_config['head_dim'],
        num_kv_heads=qwen_model_config['num_key_value_heads'],
        transpose_gating_einsum=True,
        global_base_frequency=1_000_000, # fix it so that this is used instead fo local_base_frequency
        query_pre_attn_scalar=1.0 / math.sqrt(qwen_model_config['head_dim'])
    )


@flax.struct.dataclass
class Output:
  # When `return_last_only`, `logits` is `*B V`
  logits: Float['*B L V'] | Float['*B V']
  cache: Cache | None
  hidden_states: Float['*B L D'] | Float['*B D'] | None

@flax.struct.dataclass
class _Inputs:
  embeddings: Float['B L D']
  positions: Int['B L']
  attention_mask: Bool['B L cache_length']
  inputs_mask: Bool['B L']

class QwemmaTransformer(nn.Module):
  config: QwemmaTransformerConfig
  return_last_only: bool | None = None
  dtype: jnp.dtype = jnp.bfloat16

  #def __init__(self, config):
  #  super().__init__()
  #  self.config = config

  # Keys to specify in the config which inputs to pass to the `__call__`
  # function (e.g. `tokens='batch.tokens'`).
  tokens: kontext.Key = kontext.REQUIRED
  
  def setup(self):
    self.embedder = modules.QwemmaEmbedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
    )
    self.blocks = [
        modules.QwemmaBlock(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar,
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            rope_base_frequency=self.config.global_base_frequency,
        )
        for i in range(self.config.num_layers)
    ]
    self.final_norm = modules.QwemmaRMSNorm()

  def __post_init__(self):
    super().__post_init__()

  # Calling `model.apply` on Colab makes the Kernel crash unless it is jitted.
  '''
  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'return_last_only',
          'return_hidden_states',
      ),
  )
  '''
  # The function accepts/returns aribtrary batch shape, but inside the
  # function, the batch dimension is flattened to a single dimension.
  #@_jax_utils.flatten_unflatten_batch_dim()
  #@typechecked
  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      # TODO(epot): Cleanup and simplify the API.
      positions: Int['*B L'] | None = None,
      positions_offset: Int['*B'] | None = None,
      cache: Cache | None = None,
      # During training and pre-filling, the attention mask is `*B L L`
      # When sampling (after prefilling), tokens are decoded one by one,
      # so the attention mask is `*B 1 cache_length`
      attention_mask: Bool['*B L cache_length'] | None = None,
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> Output:  # Output['*B']
    return_last_only = self._get_return_last_only(return_last_only)
    '''
        with _dtype_params.initialize_param_with_dtype(
            self.dtype,
            exclude=[
                # The multi-modal params are kept in float32.
                'vision_encoder',
                'embedder.mm_input_projection',
                'embedder.mm_soft_embedding_norm',
                # Skip the LoRA params
                'lora',
            ],
        ):
    '''

    # Encode the text tokens, eventually including the vision embeddings.
    inputs = self._encode_and_get_inputs(
        tokens=tokens,
        images=images,
        positions=positions,
        positions_offset=positions_offset,
        attention_mask=attention_mask,
    )
    del positions, attention_mask

    x = inputs.embeddings

    old_cache = cache or {}
    new_cache = {}
    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      layer_cache, x = block(
          x,
          inputs.positions,
          old_cache.get(layer_name),
          inputs.attention_mask,
      )
      new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      # TODO(epot): Use `jnp.take_along_axis`
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]

    logits = self.embedder.decode(x)

    return Output(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
    )

  @functools.partial(
      nn.jit,
      static_argnames=(
          'self',
          'batch_size',
          'dtype',
          'cache_length',
          'sharding',
      ),
  )
  def init_cache(
      self,
      *,
      batch_size: int,
      dtype: jnp.dtype[Any],
      cache_length: int,
      sharding: kd.sharding.ShardingTree | None = None,
  ) -> Cache:
    cache = self.config.init_cache(
        batch_size=batch_size,
        dtype=dtype,
        cache_length=cache_length,
    )
    return kd.sharding.with_sharding_constraint(cache, sharding)

  @typechecked
  def _encode_and_get_inputs(
      self,
      *,
      tokens: Int['B L_no_mm'],
      images: UInt8['B H W C'] | UInt8['B N H W C'] | None = None,
      attention_mask: Bool['B L_no_mm cache_length'] | None = None,
      positions: Int['B L_no_mm'] | None = None,
      positions_offset: Int['B'] | None = None,
  ) -> _Inputs:
    """Encode the text tokens, eventually including the vision embeddings."""

    # Encode the text tokens
    # Could this be optimized to filter out the `SOFT_TOKEN_PLACEHOLDER` ?
    # Currently, The placeholders are required so the mask, positions are
    # correctly computed.
    x = self.embedder.encode(tokens)
    self.sow('intermediates','qwemma.embedder.encode.out', x)

    # Compute the mask (after the extra tokens are added)
    inputs_mask = tokens != _PADDING_ID

    # Note: When `positions` and `attention_mask` are explicitly provided,
    # it's the user responsibility to correctly take into account the extra
    # tokens inserted for the images.
    # This is what the `gm.text.Sampler` implementation does.
    if positions is None:
      positions = build_positions_from_mask(inputs_mask)
      # For multi-turn, during the pre-fill phase, the positions should be
      # shifted to take into account the previous turns.
      if positions_offset is not None:
        positions += positions_offset[..., None]

    if attention_mask is None:
      if images is not None:
        bidirectional_mask = tokens == gemma_vision.TOKEN_PLACEHOLDER
      else:
        bidirectional_mask = None
      attention_mask = _attention_mask.make_causal_bidirectional_attention_mask(
          inputs_mask,
          bidirectional_mask=bidirectional_mask,
      )

    return _Inputs(
        embeddings=x,
        positions=positions,
        attention_mask=attention_mask,
        inputs_mask=inputs_mask,
    )

  def _get_return_last_only(self, return_last_only: bool | None = None) -> bool:
    """Merge `return_last_only` from the config and input."""
    # TODO(epot): Could add `default=False` to `nn.merge_param`
    if return_last_only is None and self.return_last_only is None:
      return_last_only = False
    else:
      return_last_only = nn.merge_param(
          'return_last_only', return_last_only, self.return_last_only
      )
    return return_last_only

qwen_model_config_06b = {
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": None,
  "rope_theta": 1000000,
  "sliding_window": None,
  "tie_word_embeddings": True,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": True,
  "use_sliding_window": False,
  "vocab_size": 151936
}

class QwemmaTransformer06B(QwemmaTransformer):
  config: QwemmaTransformerConfig = (
    QwemmaTransformerConfig.qwemma_06b(qwen_model_config_06b)
  )

