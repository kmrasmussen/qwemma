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
import gemma
# %%
from part6_qwemma_rmsnorm_module import t2j

# %%
from transformers.masking_utils import create_causal_mask
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

name   = "Qwen/Qwen3-0.6B"
tok    = AutoTokenizer.from_pretrained(name)
model  = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"   # 👈 force eager path
         )

ids    = tok(["hi there how are"], return_tensors="pt").input_ids



#mask_mapping = {"full_attention": mask}
#out = model(input_ids=ids, attention_mask=mask_mapping)
# %%
mask
# %%
create_causal_mask
# %%
gemma.__file__
# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(qwen_hf_model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
# %%
t = qwen_hf_model_inputs.input_ids.shape[-1]
b = 1
# %%
embeds = model.model.embed_tokens(qwen_hf_model_inputs.input_ids)
mask   = create_causal_mask(
            config          = qwen_hf_model.config,
            input_embeds    = embeds,
            attention_mask  = None,
            cache_position  = torch.arange(qwen_hf_model_inputs.input_ids.size(1)),
            past_key_values = None,
         )
# %%
mask
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
model_qwemma = Qwen3Gemma3_06B()
prompt = "hi there how are"
qwemmaprompt_input_ids = get_qwemma_tokenized(prompt)
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
treescope.show(out[1]['intermediates']['layer_0'])
# %%
treescope.show(hf_inspect_j[0])
# %%
qw_logits = out[1]['intermediates']['layer_0']['attn']['inspect.actual_attention.logits_einsum'][0]
qw_logits.shape
# %%
hf_logits = hf_inspect_j[0]['attn_intermediates']['eager_attn_intermediates']['logits_einsum'].reshape(1,8,2,4,4)
print('reshaped', hf_logits.shape)
hf_logits_perm = jnp.permute_dims(hf_logits, (0,3,1,2,4))
print('permuted', hf_logits_perm.shape)
# %%
assert jnp.allclose(qw_logits, hf_logits_perm)
# %%
assert jnp.allclose(hf_inspect_j[0]['attn_intermediates']['eager_attn_intermediates']['query_pre_attention_scalar'][0], out[1]['intermediates']['layer_0']['attn']['inspect.actual_attention.query_pre_attention_scalar'][0])
# %%
qw_logits_scaled = out[1]['intermediates']['layer_0']['attn']['inspect.actual_attention.logits_scaled'][0]
# %%
hf_logits_scaled = hf_inspect_j[0]['attn_intermediates']['eager_attn_intermediates']['logits_scaled'].reshape(1,8,2,4,4)
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
jnp.allclose(
  qw_logits_scaled, hf_logits_scaled_perm)
# %%
qw_probs = out[1]['intermediates']['layer_0']['attn']['inspect.actual_attention.probs'][0]
qw_probs.shape
# %%
hf_probs = jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['eager_attn_intermediates']['probs'], (0,2,1,3))
hf_probs.shape
# %%
hf_probs
# %%
qw_probs
# %%
assert jnp.allclose(hf_probs, qw_probs)
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['pre_attention_norm']['input'][0],hf_inspect_j[0]['input_layer_norm_intermediates']['input'])
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['pre_attention_norm']['variacne'][0],hf_inspect_j[0]['input_layer_norm_intermediates']['variance'])
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['pre_attention_norm']['rescaled_x_bfloat16'][0],hf_inspect_j[0]['input_layer_norm_intermediates']['rescaled_x_bfloat16'])
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['pre_attention_norm']['x_float32'][0],hf_inspect_j[0]['input_layer_norm_intermediates']['x_float32'])
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['pre_attention_norm']['rescaled_x'][0],hf_inspect_j[0]['input_layer_norm_intermediates']['rescaled_x'])
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['pre_attention_norm']['rescaled_x_bfloat16'][0],hf_inspect_j[0]['input_layer_norm_intermediates']['rescaled_x_bfloat16'])
# %%
assert jnp.allclose(qwen_hf_hidden_states_j[0], out[1]['intermediates']['qwemma.embedder.encode.out'][0])
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['inspect.pre_attention_norm_out'][0], hf_inspect_j[0]['input_layernorm_out']
)
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['inspect.pre_attention_norm_out'][0], hf_inspect_j[0]['input_layernorm_out'])
# %%
assert jnp.allclose(qwen_hf_hidden_states_j[0], out[1]['intermediates']['qwemma.embedder.encode.out'][0])
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['inspect.pre_attention_norm_out'][0], hf_inspect_j[0]['input_layernorm_out']
)
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.input'][0], hf_inspect_j[0]['attn_intermediates']['input']
)
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.key_proj_prenorm'][0], hf_inspect_j[0]['attn_intermediates']['k_proj_out']
)
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.key_proj_prenorm'][0], hf_inspect_j[0]['attn_intermediates']['k_proj_out']
)
# %%
#assert jnp.allclose(
#  out[1]['intermediates']['layer_0']['attn']['inspect.value_proj'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['v_proj_out'],(0,2,1,3))
#)
# %%
assert jnp.max(
  out[1]['intermediates']['layer_0']['attn']['inspect.value_proj'][0] - jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['v_proj_out'],(0,2,1,3))
) < 1e-5
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.query_norm.out'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['q_norm_out'],(0,2,1,3)))
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.q_proj_prenorm'][0], hf_inspect_j[0]['attn_intermediates']['q_proj_out']
)
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.key_norm.out'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['k_norm_out'],(0,2,1,3))
)
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['attn']['inspect.query_norm.out'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['q_norm_out'],(0,2,1,3)))
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.key_norm.out'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['k_norm_out'],(0,2,1,3))
)
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['attn']['inspect.key_roped.out'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['k_proj_roped'],(0,2,1,3)))
# %%
assert jnp.allclose(out[1]['intermediates']['layer_0']['attn']['inspect.query_roped.out'][0], jnp.permute_dims(hf_inspect_j[0]['attn_intermediates']['q_proj_roped'],(0,2,1,3)))
# %%
assert jnp.allclose(
  out[1]['intermediates']['layer_0']['attn']['inspect.o_proj.out'][0], hf_inspect_j[0]['attn_intermediates']['o_proj_out']
)