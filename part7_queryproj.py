# Part 7 - The goal of this script will be that we have a Jax module
# that is close to the Gemma codebase, ideally the same, that
# agrees with the Hf Qwen query projection

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
attention = gemma.modules.Attention(
  num_heads = qwen_model_config['num_attention_heads'],
  num_kv_heads = qwen_model_config['num_key_value_heads'],
  features =qwen_model_config['hidden_size'],
  head_dim = qwen_model_config['head_dim'],
  attn_type = AttentionType.GLOBAL,
  query_pre_attn_scalar=1.0 / math.sqrt(qwen_model_config['head_dim']),
  rope_base_frequency= qwen_model_config['rope_theta'],
  rope_scale_factor=1.0,
  attn_logits_soft_cap= None,
  sliding_window_size= None,
  use_qk_norm=True
)
attn0_params = qwemma_params['layer_0']['attn']
hf_attn0 = qwen_hf_model.model.layers[0].self_attn
# %%
#segment_pos

# %%
b = 3
d = qwen_model_config['hidden_size']
t = 5
key = jax.random.key(0)
key_x, key = jax.random.split(key)
x = jax.random.normal(key_x, (b,t,d)).astype(jnp.bfloat16)
segment_pos = jnp.tile(jnp.arange(t), (b,1)) #jnp.ones((b,t))
qwemma_attn_mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
qwemma_query_proj, qwemma_query_proj_prenorm, qwemma_key_proj, qwemma_key_proj_prenorm, qwemma_value_proj, qwemma_query_roped, qwemma_query_scaled_roped, qwemma_key_roped, qwemma_attn_output_preproj, qwemma_attn_mask, qwemma_logits, qwemma_attn_output = attention.apply({'params': attn0_params}, x, segment_pos, None, qwemma_attn_mask)
# %%
qwemma_logits
# %%
x_pt = j2t_bfloat16(x)
hf_position_embds = qwen_hf_model.model.rotary_emb(x_pt,j2t_bfloat16(segment_pos))

# Create the causal mask for the Hugging Face model.
# It expects an additive mask with 0.0 for allowed positions and a large negative number for masked ones.
# The shape should be (batch, 1, seq_len, seq_len).
hf_causal_mask = torch.ones(t, t, dtype=torch.bool, device=x_pt.device).tril()
hf_causal_mask = hf_causal_mask[None, None, :, :].expand(b, 1, t, t)
additive_hf_mask = torch.zeros_like(hf_causal_mask, dtype=x_pt.dtype)
additive_hf_mask.masked_fill_(hf_causal_mask.logical_not(), torch.finfo(x_pt.dtype).min)
# %%
1.0 / math.sqrt(qwen_model_config['head_dim'])
# %%
hf_query_proj, hf_query_proj_prenorm, hf_key_proj, hf_key_proj_prenorm, hf_value_proj, position_embeddings, hf_query_roped, hf_key_roped, hf_attn_output_preproj, hf_attn_mask, hf_logits, hf_attn_output  = hf_attn0.forward(x_pt, hf_position_embds, attention_mask=additive_hf_mask)
# %%
# %%
jnp.max(t2j(hf_attn_output) - qwemma_attn_output)
# %%
jnp.sum(t2j(hf_attn_output) != qwemma_attn_output)
# %%
qwemma_attn_output.std()
# %%
hf_attn_output_preproj.shape
# %%
hf_attn_output_preproj_reshaped = hf_attn_output_preproj.reshape(*x_pt.shape[:-1], -1).contiguous()
# %%
hf_key_proj_j = t2j(hf_key_proj)
hf_key_proj_j = jnp.einsum('abcd->acbd', hf_key_proj_j)
hf_value_proj_j = t2j(hf_value_proj)
hf_value_proj_j = jnp.einsum('abcd->acbd', hf_value_proj_j)
hf_query_proj_j = t2j(hf_query_proj)
hf_query_proj_j = jnp.einsum('abcd->acbd', hf_query_proj_j)
hf_query_proj_prenorm_j = t2j(hf_query_proj_prenorm)
# %%
# %%
qwemma_query_proj_prenorm.shape
# %%
assert jnp.allclose(hf_query_proj_prenorm_j, qwemma_query_proj_prenorm), "the query vectors before rmsnorm do not agree"
# %%
x.shape
# %%
Qw = hf_attn0.q_proj.weight.data
# %%
Qw.shape
# %%
# %%
torch.allclose(Qw, j2t_bfloat16(t2j(Qw)))
# %%
jnp.allclose(x, t2j(j2t_bfloat16(x)))
# %%
jnp.allclose(t2j(Qw), t2j(j2t_bfloat16(Qw)))
# %%
Qw.shape, x.shape
# %%
jnp.max(t2j(torch.einsum('qd,btd->btq', Qw, j2t_bfloat16(x)))-jnp.einsum('qd,btd->btq', t2j(Qw), x))
# %%
# %%
hf_query_prenorm2 = t2j(hf_attn0.q_proj(x_pt)).reshape(b,t,16,128)
hf_query_prenorm2.shape
# %%
qwemma_query_proj_prenorm.shape
# %%
jnp.argwhere(qwemma_query_proj_prenorm != hf_query_prenorm2)
# %%
# %%
hf_o_proj = t2j(hf_attn0.o_proj(hf_attn_output_preproj_reshaped))
hf_o_proj.shape
# %%

# %%
O1 = t2j(hf_attn0.o_proj.weight.data.reshape(qwen_model_config['hidden_size'], qwen_model_config['num_attention_heads'], qwen_model_config['head_dim']))
O1.shape
# %%
O2 = jnp.einsum('dnh->nhd', O1)
# %%
einsum_o_proj = jnp.einsum('BTNH,NHD->BTD', t2j(hf_attn_output_preproj), O2)
einsum_o_proj.shape
# %%
einsum_o_proj
# %%
hf_attn_output
# %%
qwemma_attn_output
# %%
hf_attn0.o_proj.weight.data.shape
# %%
# %%
hf_logits_t = t2j(hf_logits) #- qwemma_logits
# %%
hf_attn_output_j = t2j(hf_attn_output)
# %%
hf_attn_output_j - qwemma_attn_output
# %%
hf_logits_t = jnp.einsum('abcd->acbd', hf_logits_t)
# %%
jnp.max(qwemma_logits - hf_logits_t)
# %%
jnp.sum(qwemma_logits != hf_logits_t)
# %%
hf_query_roped.shape, qwemma_query_roped.shape
# %%
hf_query_roped_j = t2j(hf_query_roped)
hf_query_roped_j_reshaped1 = jnp.einsum('abcd->acbd',hf_query_roped_j)

hf_key_roped_j = t2j(hf_key_roped)
hf_key_roped_j_reshaped1 = jnp.einsum('abcd->acbd',hf_key_roped_j)
# %%
hf_query_roped_j_reshaped1.shape, qwemma_query_roped.shape
# %%
jnp.max(hf_query_roped_j_reshaped1 - qwemma_query_roped)
#jnp.allclose(hf_query_roped_j_reshaped1, qwemma_query_roped)
# %%
jnp.argwhere(hf_query_roped_j_reshaped1 != qwemma_query_roped)
# %%
hf_query_roped_j_reshaped1[1,1,13,93], qwemma_query_roped[1,1,13,93]
# %%
hf_key_roped_j_reshaped1.shape, qwemma_key_roped.shape
# %%
jnp.max(hf_key_roped_j_reshaped1 - qwemma_key_roped)
# %%
jnp.argwhere(hf_key_roped_j_reshaped1 != qwemma_key_roped)
# %%
# %%

# %%
hf_query_proj_j.shape
# %%
qwemma_query_proj.shape
# %%
print('query postnorm')
max_diff = jnp.max(qwemma_query_proj - hf_query_proj_j)
print('max diff', max_diff)
# %%
n_nonzero_diffs = (qwemma_query_proj != hf_query_proj_j).sum()
print('number of non-zero diffs', n_nonzero_diffs)
# %%

# %%
hf_key_proj_prenorm.shape, qwemma_key_proj_prenorm.shape
# %%
print('key proj max diff', jnp.max(t2j(hf_key_proj_prenorm) - qwemma_key_proj_prenorm))
# %%
hf_value_proj_j.shape, qwemma_value_proj.shape
# %%
#jnp.sum(hf_value_proj_j != qwemma_value_proj)
# %%
print('value proj max diff', jnp.max(hf_value_proj_j - qwemma_value_proj))
# %%
hf_key_proj_j.shape, qwemma_key_proj.shape
# %%
jnp.max(hf_key_proj_j - qwemma_key_proj)
# %%
# %%

# %%
# %%
hf_attn0.forward
# %%
hf_attn_output_preproj.shape
# %%
qwemma_attn_output_preproj.shape
# %%
hf_attn_output_preproj_j = t2j(hf_attn_output_preproj)
# %%
attn_diffs = jnp.max(qwemma_attn_output_preproj - hf_attn_output_preproj_j)
# %%
hf_attn_mask.shape
# %%
print('aatn max diff', attn_diffs)
n_attn_diffs = jnp.sum(qwemma_attn_output_preproj != hf_attn_output_preproj_j)
print('n attn diffs', n_attn_diffs)
# %%
attn_subtraction = qwemma_attn_output_preproj - hf_attn_output_preproj_j
# %%
attn_subtraction.shape
# %%
import matplotlib.pyplot as plt
# %%
plt.matshow(attn_subtraction[1,4])
# %%
plt.hist(attn_subtraction.reshape(-1), bins=50)
# %%
hf_attn_output
# %%
qwemma_attn_output
# %%
einsum_o_proj2 = jnp.einsum('BTNH,NHD->BTD', t2j(hf_attn_output_preproj), O2)
einsum_o_proj.shape