# Part 5 - The goal of this script will be to ensure and verify 
# that the initial parts of the forward pass up to the beginning of the
# first Transformer block will match Huggingface Qwen3

# Step 1
# 
# Fetch the input to Transformer block 0
# Be able to print at random places in the HF-Qwen forward pass

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import jax
from part4_preliminary_qwemma_forward_pass import *
import torch
import gemma



# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(qwen_hf_model_name)
# %%
try:
  qwen_hf_outputs = qwen_hf_model(**qwen_hf_model_inputs, output_hidden_states=True)
  qwen_hf_R_s0 = jax.array(qwen_hf_outputs.hidden_states[0].detach())
  qwen_hf_model_inputs.input_ids.shape
  qwen_hf_R_s0.shape
  #print('qwen_hfmodel Rs0 from output', qwen_hf_R_s0[0,0,0])
except Exception as e:
  pass

print('transformers dep path', transformers.__file__)
print('gemma dep path', gemma.__file__)

# Step 2
# For Qwemma do the same

# %%

out_qwemma = do_jax_forward_pass(qwemmaprompt_input_ids, model_qwemma, qwemma_params)
# %%
qwen_hf_model.model.layers[0].input_layernorm.weight.data
# %%
qwemma_params['layer_0']['pre_attention_norm']['scale']
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
