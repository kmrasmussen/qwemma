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
# %%
qwemma_params = get_qwengemma06b_params()
# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(qwen_hf_model_name, torch_dtype=torch.bfloat16)
# %%
# %%
attention = gemma.modules.Attention(
  num_heads = qwen_model_config['num_attention_heads'],
  num_kv_heads = qwen_model_config['num_key_value_heads'],
  features =qwen_model_config['hidden_size'],
  head_dim = qwen_model_config['head_dim'],
  attn_type = AttentionType.GLOBAL,
  query_pre_attn_scalar= 1.,
  rope_base_frequency= qwen_model_config['rope_theta'],
  rope_scale_factor= None,
  attn_logits_soft_cap= None,
  sliding_window_size= None,
  use_qk_norm=True
)
attn0_params = qwemma_params['layer_0']['attn']
hf_attn0 = qwen_hf_model.model.layers[0].self_attn
'''
hf_qproj_weights = hf_attn0.q_proj.weight.data
hf_qproj_weights_reshaping1 = hf_qproj_weights.T.reshape(qwen_model_config['hidden_size'], qwen_model_config['num_attention_heads'],qwen_model_config['head_dim'])
hf_qproj_weights_reshaping1.shape
hf_qproj_weights_reshaping2 = torch.einsum('DNH->NDH', hf_qproj_weights_reshaping1)
hf_qproj_weights_reshaping2.shape
hf_qproj_weights_j = t2j(hf_qproj_weights_reshaping2)
attn0_params['q_einsum']['w'] = hf_qproj_weights_j
'''
# %%
b = 3
d = qwen_model_config['hidden_size']
t = 4
key = jax.random.key(0)
key_x, key = jax.random.split(key)
x = jax.random.normal(key_x, (b,t,d)).astype(jnp.bfloat16)
segment_pos = jnp.ones((b,t))
qwemma_query_proj, qwemma_query_proj_prenorm, qwemma_key_proj, qwemma_key_proj_prenorm, qwemma_value_proj = attention.apply({'params': attn0_params}, x, segment_pos, None, None)
# %%
x_pt = j2t_bfloat16(x)
hf_query_proj, hf_query_proj_prenorm, hf_key_proj, hf_key_proj_prenorm, hf_value_proj  = hf_attn0.forward(x_pt, None, attention_mask=None)
hf_key_proj_j = t2j(hf_key_proj)
hf_value_proj_j = t2j(hf_value_proj)
hf_value_proj_j = jnp.einsum('abcd->acbd', hf_value_proj_j)
hf_query_proj_j = t2j(hf_query_proj)
hf_query_proj_j = jnp.einsum('abcd->acbd', hf_query_proj_j)
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
jnp.max(t2j(hf_key_proj_prenorm) - qwemma_key_proj_prenorm)
# %%
hf_value_proj_j.shape, qwemma_value_proj.shape
# %%
jnp.sum(hf_value_proj_j != qwemma_value_proj)
# %%
jnp.max(hf_value_proj_j - qwemma_value_proj)
# %%
hf_value_proj_j - qwemma_value_proj
# %%
hf_key_proj_prenorm_j = t2j(hf_key_proj_prenorm)
hf_key_proj_prenorm_j = jnp.einsum('abcd->acbd', hf_key_proj_prenorm_j)
# %%
hf_key_proj_prenorm_j.shape, qwemma_key_proj_prenorm.shape
# %%
# %%
hf_key_proj_prenorm.shape
# %%

hf_attn = qwen_hf_model.model.layers[0].self_attn
hf_kproj_weights = hf_attn.k_proj.weight.data
hf_qproj_weights = hf_attn.q_proj.weight.data
hf_vproj_weights = hf_attn.v_proj.weight.data
#hf_kproj_weights_reshaping1 = hf_qproj_weights.T.reshape(qwen_model_config['hidden_size'], qwen_model_config['num_attention_heads'],qwen_model_config['head_dim'])
#hf_kproj_weights_reshaping2 = torch.einsum('DNH->NDH', hf_qproj_weights_reshaping1)
#hf_kproj_weights_j = t2j(hf_qproj_weights_reshaping2)
# %%
hf_kproj_weights.shape, hf_qproj_weights.shape
# %%
hf_kv_weights = t2j(torch.stack([hf_kproj_weights, hf_vproj_weights]))
hf_kv_weights.shape
# %%
# CKDH
hf_kv_weights_reshaped1 = hf_kv_weights.reshape(2, qwen_model_config['num_key_value_heads'], qwen_model_config['head_dim'], qwen_model_config['hidden_size'])
hf_kv_weights_reshaped1.shape
hf_kv_weights_reshaped2 = jnp.einsum('cnhd->cndh', hf_kv_weights_reshaped1)
# %%
# %%
attn0_params['kv_einsum']['w'].shape
# %%
hf_kv_weights_reshaped2.shape
# %%
#hf_key_proj_prenorm
hf_kv_einsum = jnp.einsum('BSD,CKDH->CBSKH',x,hf_kv_weights_reshaped2)
# %%
hf_kv_einsum.shape
# %%
hf_kv_einsum[0].shape
# %%
qwemma_key_proj.shape
# %%
jnp.max(hf_kv_einsum[0] - t2j(hf_key_proj_prenorm))
# %%
jnp.sum(hf_kv_einsum[0] != t2j(hf_key_proj_prenorm))
# %%
