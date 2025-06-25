# Part 9 - now that we have the attention module working with max diff of 0.01
# we try to get the remaining part of the transformer block

# %%
%load_ext autoreload
%autoreload 2
# %%
import gemma
import gemma.modules
import jax
import torch
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
import treescope
from part3_qwen_params_dict import qwen_model_config, get_qwengemma06b_params
from part6_qwemma_rmsnorm_module import t2j, j2t_bfloat16
from gemma.modules import AttentionType
import math
# %%
import os, torch, jax, jax.numpy as jnp, numpy as np
torch.set_num_threads(1)
os.environ["JAX_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"   # optional extra safety
jax.config.update("jax_default_matmul_precision", "bfloat16")  # or "float32"
jax.config.update("jax_enable_x64", True)
# %%
qwemma_params = get_qwengemma06b_params()
# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(qwen_hf_model_name, torch_dtype=torch.bfloat16)
# %%
block = gemma.modules.Block(
  num_heads = qwen_model_config['num_attention_heads'],
  num_kv_heads = qwen_model_config['num_key_value_heads'],
  embed_dim =qwen_model_config['hidden_size'],
  hidden_dim=qwen_model_config['intermediate_size'],
  head_dim = qwen_model_config['head_dim'],
  use_post_attn_norm=True,
  use_post_ffw_norm=False,
  transpose_gating_einsum=True,
  attn_type = AttentionType.GLOBAL,
  query_pre_attn_scalar=1.0 / math.sqrt(qwen_model_config['head_dim']),
  rope_base_frequency= qwen_model_config['rope_theta'],
  rope_scale_factor=1.0,
  attn_logits_soft_cap= None,
  sliding_window_size= None,
  use_qk_norm=True
)
# %%
import treescope
treescope.show(qwemma_params)
# %%
b = 3
d = qwen_model_config['hidden_size']
t = 5
key = jax.random.key(0)
key_x, key = jax.random.split(key)
x = jax.random.normal(key_x, (b,t,d)).astype(jnp.bfloat16)
segment_pos = jnp.tile(jnp.arange(t), (b,1)) #jnp.ones((b,t))
qwemma_attn_mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
qw_block_out, qw_intermediates0 = block.apply({'params': qwemma_params['layer_0']}, x, segment_pos, None, qwemma_attn_mask, capture_intermediates=True)
qw_intermediates = qw_intermediates0['intermediates']
# %%

# %%
# %%
x_pt = j2t_bfloat16(x)

hf_causal_mask = torch.ones(t, t, dtype=torch.bool, device=x_pt.device).tril()
hf_causal_mask = hf_causal_mask[None, None, :, :].expand(b, 1, t, t)
additive_hf_mask = torch.zeros_like(hf_causal_mask, dtype=x_pt.dtype)
additive_hf_mask.masked_fill_(hf_causal_mask.logical_not(), torch.finfo(x_pt.dtype).min)
# %%
hf_block0 = qwen_hf_model.model.layers[0]
hf_position_embds = qwen_hf_model.model.rotary_emb(x_pt,j2t_bfloat16(segment_pos))
# %%
len(hf_position_embds)
# %%
hf_block_out, hf_intermediates  = hf_block0.forward(
  hidden_states=x_pt,
  attention_mask=additive_hf_mask,
  position_embeddings=hf_position_embds)
# %%
treescope.show(qw_intermediates)
# %%
treescope.show(hf_intermediates)
# %%
hf_intermediates_j = jax.tree_util.tree_map(lambda tensor: t2j(tensor), hf_intermediates)
treescope.show(hf_intermediates_j)
# %%
hf_intermediates_j['input_layernorm_out']
# %%
treescope.show(qw_intermediates)
# %%
assert jnp.allclose(
  qw_intermediates['pre_attention_norm']['__call__'][0], 
  hf_intermediates_j['input_layernorm_out'],
  rtol=1-8)
# %%
assert jnp.allclose(
  hf_intermediates_j['attn_intermediates']['q_proj_out'],
  qw_intermediates['attn']['q_einsum']['__call__'][0],
  rtol=1e-2)
# %%
assert jnp.max(
  hf_intermediates_j['attn_intermediates']['o_proj_out'] - qw_intermediates['attn']['attn_vec_einsum']['__call__'][0]
) < 0.005
# %%
assert jnp.max(qw_intermediates['post_attention_norm']['__call__'][0] -
             hf_intermediates_j['post_attention_layernorm_out']) < 0.008
# %%
#hf_mlp_out, hf_mlp_intermediates = qwen_hf_model.model.layers[0].mlp.forward_intermediates(hf_intermediates['post_attention_layernorm_out'])
# %%
#treescope.show(hf_mlp_intermediates)
# %%
assert jnp.max(
  hf_intermediates_j['mlp_intermediates']['gate_proj_out'] - qw_intermediates['mlp']['__call__'][0][:,:,0,:]
  ) < 0.02
# %%
assert jnp.max(
  hf_intermediates_j['mlp_intermediates']['up_proj_out'] - qw_intermediates['mlp']['__call__'][0][:,:,1,:]
  ) < 0.01
# %%
assert jnp.max(
  hf_intermediates_j['mlp_out'] - qw_intermediates['mlp']['__call__'][1]
  ) < 0.004
# %%

# %%
hf_block_out[0]
# %%
treescope.show(qw_block_out[2])

# %%
treescope.show(hf_intermediates_j)
# %%
assert jnp.max(
  hf_intermediates_j['attn_out_added'] - qw_block_out[2]['attn_added_out']
  ) < 0.02
# %%
assert jnp.max(
  hf_intermediates_j['mlp_out_added'] - qw_block_out[2]['mlp_added_out']
) < 0.02
# %%
assert jnp.max(
  qw_block_out[1] - t2j(hf_block_out[0])
  ) < 0.02
# %%
