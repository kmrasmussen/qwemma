# Part 5 - The goal of this script will be to ensure and verify 
# that the initial parts of the forward pass up to the beginning of the
# first Transformer block will match Huggingface Qwen3

# Step 1
# 
# Fetch the input to Transformer block 0
# Be able to print at random places in the HF-Qwen forward pass
# %%
%load_ext autoreload
%autoreload 2
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import jax
from part4_preliminary_qwemma_forward_pass import *
import torch
from part6_qwemma_rmsnorm_module import t2j, j2t_bfloat16
from transformers.masking_utils import create_causal_mask
# %%
def kdiff(a,b):
  diff1 = jnp.max(jnp.abs(a-b))
  diff2 = jnp.max(jnp.abs(b-a))
  assert diff1 == diff2
  allclose = jnp.allclose(a,b)
  return diff1.item(), allclose.item()
# %%
os.environ["JAX_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"   # optional extra safety
jax.config.update("jax_default_matmul_precision", "bfloat16")  # or "float32"
jax.config.update("jax_enable_x64", True)

# %%
# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(
  qwen_hf_model_name, 
  torch_dtype=torch.bfloat16, 
  attn_implementation="eager")
# %%
t = qwen_hf_model_inputs.input_ids.shape[-1]
b = 1
# %%
embeds = qwen_hf_model.model.embed_tokens(qwen_hf_model_inputs.input_ids)
mask   = create_causal_mask(
            config          = qwen_hf_model.config,
            input_embeds    = embeds,
            attention_mask  = None,
            cache_position  = torch.arange(qwen_hf_model_inputs.input_ids.size(1)),
            past_key_values = None,
         )
# %%
qwen_hf_outputs = qwen_hf_model(qwen_hf_model_inputs.input_ids, output_hidden_states=True, attention_mask=mask)
# %%
qwen_hf_model_inputs
# %%
qwen_hf_model_inputs.input_ids
# %%
hf_embds = qwen_hf_model.model.embed_tokens(qwen_hf_model_inputs.input_ids)
# %%
qwen_hf_hidden_states_j = jax.tree_util.tree_map(lambda t: t2j(t), qwen_hf_outputs.hidden_states)
# %%
# %%
prompt = "hi there how are"
qwemmaprompt_input_ids = get_qwemma_tokenized(prompt)
# %%
model_qwemma = Qwen3Gemma3_06B()

qwemma_params = get_qwengemma06b_params()
out = model_qwemma.apply(
    {'params': qwemma_params},
    tokens=qwemmaprompt_input_ids,
    return_last_only=True,  # Only predict the last token
    capture_intermediates=True
)
# %%
hf_inspect_j = jax.tree_util.tree_map(lambda t: t2j(t), qwen_hf_outputs.layer_intermediates_list, )
# %%
s = 0
#%%
treescope.show(out[1]['intermediates'][f'layer_{s}'])
# %%
treescope.show(hf_inspect_j[s])
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['inspect.input'][0], hf_inspect_j[s]['input']
  )
# %%
assert jnp.max(
  out[1]['intermediates'][f'layer_{s}']['inspect.input'][0] - hf_inspect_j[s]['input']
  ) < .004
# %%
out[1]['intermediates']['layer_0']['pre_attention_norm']['input'][0].shape
# %%
hf_inspect_j[s]['paln_intermediates']['input'].shape
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['input'][0],
  hf_inspect_j[s]['paln_intermediates']['input'])
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32'][0],
  hf_inspect_j[s]['paln_intermediates']['x_float32'])
# %%
print('x diff', kdiff(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['input'][0],
  hf_inspect_j[s]['paln_intermediates']['input'])
)
print('x_float32 diff', kdiff(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32'][0],
  hf_inspect_j[s]['paln_intermediates']['x_float32']))
print('x_float32_squared diff', kdiff(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32_squared'][0],hf_inspect_j[s]['paln_intermediates']['x_float32_squared']))
print('variance diff', kdiff(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['variacne'][0],hf_inspect_j[s]['paln_intermediates']['variance']))
print('rescaled_x diff', kdiff(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['rescaled_x'][0],hf_inspect_j[s]['paln_intermediates']['rescaled_x']))
print('rescaled_x_bfloat16 diff', kdiff(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['rescaled_x_bfloat16'][0],hf_inspect_j[s]['paln_intermediates']['rescaled_x_bfloat16']))
# %%
print('attn added out', kdiff(out[1]['intermediates'][f'layer_{s}']['inspect.attn_added_out'][0],
 hf_inspect_j[s]['attn_out_added']))
# %%
print('mlp out out', kdiff(out[1]['intermediates'][f'layer_{s}']['inspect.mlp_out'][0],
 hf_inspect_j[s]['mlp_out']))
# %%
# %%
x_float32 = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32'][0]
va = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32_squared'][0]
vb = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['variacne'][0]
va2 = jnp.square(va)
variance1 = jnp.mean(
      jnp.square(x_float32),
      axis=-1,
      keepdims=True,
      dtype=jnp.float32
)
variance2 = jnp.mean(
      va,
      axis=-1,
      keepdims=True,
      dtype=jnp.float32
)
# %%

# %%
kdiff(out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32'][0],
  hf_inspect_j[s]['paln_intermediates']['x_float32'])
# %%
kdiff(out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32_squared'][0],
  hf_inspect_j[s]['paln_intermediates']['x_float32_squared'])
# %%

# %%
kdiff(variance1, variance2)
# %%
jnp.max(jnp.abs(variance1 - variance2))
# %%
jnp.max(jnp.abs(vb - variance1))
# %%

# %%
hv = hf_inspect_j[s]['paln_intermediates']['variance']
# %%
kdiff(hv, vb)

# %%
kdiff(hv, variance1)
# %%
vat = j2t
# %%
jnp.max(hv - vb)
# %%
jnp.max(out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['variacne'][0]-hf_inspect_j[s]['paln_intermediates']['variance'])
# %%
bob1 = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['variacne'][0]
jnp.max(bob1 - vb)
# %%
bob2 = hf_inspect_j[s]['paln_intermediates']['variance']
jnp.max(bob2 - hv)
# %%
jnp.max(bob1 - bob2)
# %%
jnp.max(hv - vb)
# %%
jnp.max(vb - hv)
# %%
jnp.max(jnp.abs(hv - vb))
# %%
jnp.max(jnp.abs(vb - hv))
# %%
a = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['input'][0]
# %%
a.dtype
# %%
a2 = t2j(j2t_bfloat16(a).to(torch.float32))
# %%
b = hf_inspect_j[s]['paln_intermediates']['input']
b.dtype
# %%
b2 = b.astype(jnp.float32)
# %%
jnp.max(a2 - b2)
# %%
c2 = a.astype(jnp.float32)
# %%
jnp.max(a2 - c2)
# %%
d = qwen_hf_outputs.layer_intermediates_list[s]['paln_intermediates']['input']
d.dtype
# %%
d2 = d.to(torch.float32)
# %%
d2j = t2j(d2)
d2j.dtype
# %%
jnp.max(a.astype(jnp.float32) - d2j)
# %%
e = qwen_hf_outputs.layer_intermediates_list[s]['paln_intermediates']['x_float32']
e.dtype
# %%
ej = t2j(e)
# %%
jnp.max(ej - a.astype(jnp.float32))
# %%
a3 = jnp.asarray(a, jnp.float32)
# %%
s
# %%
f = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['x_float32'][0]
# %%
jnp.max(ej - f)
# %%
g = a.astype(jnp.float32)
# %%
jnp.max(ej - g)
# %%
u = out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['input'][0].astype(jnp.float32)
# %%
jnp.max(ej - u)
# %%
jnp.max(f-g)
# %%
jnp.max(f-a3)
# %%
jnp.max(a3-ej)
# %%
a.dtype
# %%
from gemma.layers import QwemmaRMSNorm
# %%
qwrmsnorm = QwemmaRMSNorm()
# %%
qwrmsout = qwrmsnorm.apply(
  {'params': qwemma_params['layer_0']['post_attention_norm']}, 
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['input'][0],
  capture_intermediates=True)[1]['intermediates']
#%%
treescope.show(qwrmsout)
# %%
jnp.max(qwrmsout['x_float32'][0] - ej)
# %%
t2j(qwen_hf_outputs.layer_intermediates_list[s]['paln_intermediates']['variance']) - qwrmsout['variacne'][0]
# %%
# %%
from modeling_qwemma import QwemmaBlock
# %%
qwen_model_config
# %%
from qwemma_transformer import *
# %%
from part3_qwen_params_dict import qwen_model_config 
# %%
qwemma_config = QwemmaTransformerConfig.qwemma_06b(qwen_model_config)
# %%
qwemma_transformer = QwemmaTransformer06B()
# %%
qwemma_transformer
# %%
qwt_out = qwemma_transformer.apply(   
  {'params': qwemma_params},
  tokens=qwemmaprompt_input_ids,
  return_last_only=True,  # Only predict the last token
  capture_intermediates=True)
# %%
treescope.show(out)
# %%
treescope.show(qwt_out)
# %%
jnp.max(out[1]['intermediates']['layer_0']['post_attention_norm']['x_float32'][0]  - qwt_out[1]['intermediates']['layer_0']['post_attention_norm']['x_float32'][0])
# %%
# %%
j = qwt_out[1]['intermediates']['layer_0']['post_attention_norm']['input'][0]
# %%
k = qwt_out[1]['intermediates']['layer_0']['post_attention_norm']['input'][0].astype(jnp.float32)
# %%
j.dtype, k.dtype
# %%
qwt_out[1]['intermediates']['layer_0']['post_attention_norm']['x_float32'][0].dtype
# %%
# %%
jnp.max(out[1]['intermediates']['layer_0']['post_attention_norm']['x_float32'][0]  - k)
# %%
jnp.max(ej - k)
# %%
jnp.max(qwt_out[1]['intermediates']['layer_0']['post_attention_norm']['x_float32'][0]  - k)
# %%
jnp.max(qwt_out[1]['intermediates']['layer_0']['post_attention_norm']['x_float32'][0]  - ej)
# %%
hf_inspect_j[s]['input']
# %%
# %%
    num_layers=_NUM_LAYERS_GEMMA3_1B,
        final_logit_softcap=None,
        num_embed=qwen_model_config['vocab_size'],
        embed_dim=qwen_model_config['hidden_size'],
        hidden_dim=qwen_model_config['intermediate_size'],
        num_heads=qwen_model_config['num_attention_heads'],
        head_dim=qwen_model_config['head_dim'],
        num_kv_heads=qwen_model_config['num_key_value_heads'],
        use_post_attn_norm=True,
        use_post_ffw_norm=False,
        use_qk_norm=True,
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_1B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=512,
        transpose_gating_einsum=True,
        local_base_frequency=1_000_000,
        global_base_frequency=1_000_000, # fix it so that this is used instead fo local_base_frequency
        vision_encoder=None,
        qwemma_setting_rescale_embeddings=False,
        query_pre_attn_scalar=1.0 / math.sqrt(qwen_model_config['head_dim'])
    
# %%
qwblock = QwemmaBlock(
  num_heads = qwen_model_config['num_attention_heads'],
  num_kv_heads = qwen_model_config['num_key_value_heads'],
  embed_dim = qwen_model_config['hidden_size'],
  head_dim = qwen_model_config['head_dim'],
  hidden_dim = qwen_model_config['intermediate_size'],
  query_pre_attn_scalar = 1.0 / math.sqrt(qwen_model_config['head_dim']),
  transpose_gating_einsum = True,
  rope_base_frequency = 1_000_000,
  rope_scale_factor = 1.
)
# %%

# %%
segment_pos = jnp.tile(jnp.arange(t), (1,1)) #jnp.ones((b,t))
# %%
qwemma_attn_mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
qw_block_out, qw_intermediates0 = qwblock.apply(
  {'params': qwemma_params['layer_0']}, 
  out[1]['intermediates'][f'layer_{s}']['inspect.input'][0], 
  segment_pos, 
  None, 
  qwemma_attn_mask, 
  capture_intermediates=True)
# %%
treescope.show(qw_intermediates0)
# %%
treescope.show(out[1]['intermediates'][f'layer_{s}'])
# %%
N = qw_intermediates0['intermediates']
O = out[1]['intermediates'][f'layer_{s}']
# %%
jax.tree_util.tree_map(lambda a,b: jnp.max(a-b), qw_intermediates0['intermediates'], out[1]['intermediates'][f'layer_{s}'])
# %%
jnp.max(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['normed_inputs'][0]-hf_inspect_j[s]['paln_intermediates']['normed_inputs'])
# %%
mlp_out_diff = jnp.max(
  out[1]['intermediates'][f'layer_{s}']['inspect.mlp_out'][0] - hf_inspect_j[s]['mlp_out']
)
assert mlp_out_diff < 0.004, mlp_out_diff
# %%
assert jnp.max(
  out[1]['intermediates'][f'layer_{s}']['inspect.block_out'][0] - hf_inspect_j[s]['mlp_out_added']
) < 0.004
# %%
jnp.max(
  out[1]['intermediates'][f'layer_{s}']['inspect.block_out'][0] - hf_inspect_j[s]['mlp_out_added']
) < 0.004
# %%
jnp.max(
  out[1]['intermediates'][f'layer_{s}']['post_attention_norm']['rescaled_x'][0]- hf_inspect_j[s]['paln_intermediates']['rescaled_x'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['inspect.attn_out'][0],hf_inspect_j[s]['attn_out'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['inspect.attn_added_out'][0],hf_inspect_j[s]['attn_out_added'])
# %%
# %%
jnp.max(out[1]['intermediates'][f'layer_{s}']['inspect.post_attention_norm_out'][0]-hf_inspect_j[s]['post_attention_layernorm_out'])
# %%
paln_hf = t2j(qwen_hf_model.model.layers[0].post_attention_layernorm.weight.data)
# %%
paln_qwemma = qwemma_params[f'layer_{s}']['post_attention_norm']['scale']
# %%
jnp.allclose(paln_hf,paln_qwemma)
# %%
qw_logits = out[1]['intermediates'][f'layer_{s}']['attn']['inspect.actual_attention.logits_einsum'][0]
qw_logits.shape
# %%
hf_logits = hf_inspect_j[s]['attn_intermediates']['eager_attn_intermediates']['logits_einsum'].reshape(1,8,2,4,4)
print('reshaped', hf_logits.shape)
hf_logits_perm = jnp.permute_dims(hf_logits, (0,3,1,2,4))
print('permuted', hf_logits_perm.shape)
# %%
assert jnp.allclose(qw_logits, hf_logits_perm)
# %%
assert jnp.allclose(hf_inspect_j[s]['attn_intermediates']['eager_attn_intermediates']['query_pre_attention_scalar'][0], out[1]['intermediates'][f'layer_{s}']['attn']['inspect.actual_attention.query_pre_attention_scalar'][0])
# %%
qw_logits_scaled = out[1]['intermediates'][f'layer_{s}']['attn']['inspect.actual_attention.logits_scaled'][0]
# %%
hf_logits_scaled = hf_inspect_j[s]['attn_intermediates']['eager_attn_intermediates']['logits_scaled'].reshape(1,8,2,4,4)
print('reshaped', hf_logits_scaled.shape)
hf_logits_scaled_perm = jnp.permute_dims(hf_logits_scaled, (0,3,1,2,4))
print('permuted', hf_logits_scaled_perm.shape)
# %%
qw_logits_scaled.shape
# %%
qw_logits.dtype, hf_logits_perm.dtype
# %%
hf_logits_scaled_perm.dtype,hf_logits_scaled_perm.dtype
# %%
qw_logits_scaled.dtype 
# %%
# %%
qw_probs = out[1]['intermediates'][f'layer_{s}']['attn']['inspect.actual_attention.probs'][0]
qw_probs.shape
# %%
hf_probs = jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['eager_attn_intermediates']['probs'], (0,2,1,3))
hf_probs.shape
# %%
hf_probs
# %%
qw_probs
# %%
assert jnp.allclose(hf_probs, qw_probs)
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['pre_attention_norm']['input'][0],hf_inspect_j[s]['input_layer_norm_intermediates']['input'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['pre_attention_norm']['variacne'][0],hf_inspect_j[s]['input_layer_norm_intermediates']['variance'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['pre_attention_norm']['rescaled_x_bfloat16'][0],hf_inspect_j[s]['input_layer_norm_intermediates']['rescaled_x_bfloat16'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['pre_attention_norm']['x_float32'][0],hf_inspect_j[s]['input_layer_norm_intermediates']['x_float32'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['pre_attention_norm']['rescaled_x'][0],hf_inspect_j[s]['input_layer_norm_intermediates']['rescaled_x'])
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['pre_attention_norm']['rescaled_x_bfloat16'][0],hf_inspect_j[s]['input_layer_norm_intermediates']['rescaled_x_bfloat16'])
# %%
assert jnp.allclose(qwen_hf_hidden_states_j[0], out[1]['intermediates']['qwemma.embedder.encode.out'][0])
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['inspect.pre_attention_norm_out'][0], hf_inspect_j[s]['input_layernorm_out']
)
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['inspect.pre_attention_norm_out'][0], hf_inspect_j[s]['input_layernorm_out'])
# %%
assert jnp.allclose(qwen_hf_hidden_states_j[0], out[1]['intermediates']['qwemma.embedder.encode.out'][0])
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['inspect.pre_attention_norm_out'][0], hf_inspect_j[s]['input_layernorm_out']
)
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.input'][0], hf_inspect_j[s]['attn_intermediates']['input']
)
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.key_proj_prenorm'][0], hf_inspect_j[s]['attn_intermediates']['k_proj_out']
)
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.key_proj_prenorm'][0], hf_inspect_j[s]['attn_intermediates']['k_proj_out']
)
# %%
#assert jnp.allclose(
#  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.value_proj'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['v_proj_out'],(0,2,1,3))
#)
# %%
assert jnp.max(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.value_proj'][0] - jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['v_proj_out'],(0,2,1,3))
) < 1e-5
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.query_norm.out'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['q_norm_out'],(0,2,1,3)))
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.q_proj_prenorm'][0], hf_inspect_j[s]['attn_intermediates']['q_proj_out']
)
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.key_norm.out'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['k_norm_out'],(0,2,1,3))
)
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['attn']['inspect.query_norm.out'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['q_norm_out'],(0,2,1,3)))
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.key_norm.out'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['k_norm_out'],(0,2,1,3))
)
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['attn']['inspect.key_roped.out'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['k_proj_roped'],(0,2,1,3)))
# %%
assert jnp.allclose(out[1]['intermediates'][f'layer_{s}']['attn']['inspect.query_roped.out'][0], jnp.permute_dims(hf_inspect_j[s]['attn_intermediates']['q_proj_roped'],(0,2,1,3)))
# %%
assert jnp.allclose(
  out[1]['intermediates'][f'layer_{s}']['attn']['inspect.o_proj.out'][0], hf_inspect_j[s]['attn_intermediates']['o_proj_out']
)
# %%
