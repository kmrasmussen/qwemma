# Part 10 - the goal of this script is to look at how the max diff
# increases along depth, in part9 we had max diff of 0.02 for a single
# block with standard normal distribution on inputs
# %%
%load_ext autoreload
%autoreload 2

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import jax
from part4_preliminary_qwemma_forward_pass import *
from part6_qwemma_rmsnorm_module import t2j, j2t_bfloat16
import torch
import gemma
import treescope
# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(qwen_hf_model_name)
# %%
qwen_hf_outputs = qwen_hf_model(**qwen_hf_model_inputs, output_hidden_states=True)
# %%
treescope.show(qwen_hf_outputs['hidden_states'])
# %%
hf_hiddens_j = jax.tree_util.tree_map(lambda tensor: t2j(tensor), qwen_hf_outputs['hidden_states'])
treescope.show(hf_hiddens_j)
# %%
out_qwemma = do_jax_forward_pass(qwemmaprompt_input_ids, model_qwemma, qwemma_params)
# %%
treescope.show(out_qwemma[1])
# %%
out_qwemma[1]['intermediates']['layer_0']['__call__'][0][1]
# %%
treescope.show(hf_hiddens_j)
# %%
len(hf_hiddens_j)
# %%
# %%
qwen_hf_model.model.layers[0].input_layernorm.weight.data
# %%
qwemma_params['layer_0']['pre_attention_norm']['scale']
# %%

# %%
#print(out_qwemma)
# %%
#treescope.show(out_qwemma[1])
# %%
#treescope.show(out_qwemma[1]['intermediates'])
# %%
#qwemma_R_s0 = out_qwemma[1]['intermediates']['layer_0']['__call__'][0][1]
# %%
#qwen_hf_R_s1 = jax.dlpack.from_dlpack(qwen_hf_outputs.hidden_states[1].detach())

# %%
#qwen_hf_R_s1
# %%
# Step 3
# Gemma-codebase rescales the embeddings
# modules.py ~line100: x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
# Now added a setting

# Step 4 Pre-attention normalization should agree
