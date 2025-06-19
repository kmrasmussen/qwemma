# The goal of this script is to load the HuggingFace Qwen3 and take the weights
# and put them in the same format as the params object for Gemma

# %%
import treescope
import gemma
from gemma import gm
from etils import epath 
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
import jax
import jax.numpy as jnp
import torch
def t2j(t: torch.Tensor):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))

#
print('gemma file', gemma.__file__)
# %%
qwen_model_config = {
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
# %%
#local_params_epath_str = './params/gemma3_1b_it' # This was for Orbax
#local_params_absolute_epath = epath.Path(local_params_epath_str).resolve() # Keep for cleanup if needed
#gemma_params = gm.ckpts.load_params(local_params_absolute_epath )
# %%
#treescope.show(gemma_params)
# %%
def get_q_einsum_for_layer(current_layer):
    hf_attn = current_layer.self_attn
    hf_qproj_weights = hf_attn.q_proj.weight.data
    hf_qproj_weights_reshaping1 = hf_qproj_weights.T.reshape(qwen_model_config['hidden_size'], qwen_model_config['num_attention_heads'],qwen_model_config['head_dim'])
    hf_qproj_weights_reshaping2 = torch.einsum('DNH->NDH', hf_qproj_weights_reshaping1)
    hf_qproj_weights_j = t2j(hf_qproj_weights_reshaping2)
    return hf_qproj_weights_j

def get_qwengemma06b_params():
    qwen_model_name = "Qwen/Qwen3-0.6B"
    qwen_hf_model = AutoModelForCausalLM.from_pretrained(
        qwen_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},      # every sub-module → CPU
        low_cpu_mem_usage=False,     # **materialise** weights immediately
        offload_state_dict=False,
    )

    # embedder.input_embedding 
    qwen_hf_E_in = qwen_hf_model.model.embed_tokens._parameters['weight'].data
    qwen_E_in_jax = t2j(qwen_hf_E_in.cpu().detach())
    # final_norm.scale
    qwen_gemma_final_norm_scale = t2j(qwen_hf_model.model.norm.weight.data)
    qwengemma_params = {}
    qwengemma_params['embedder'] = {
        'input_embedding': qwen_E_in_jax
    }
    qwengemma_params['final_norm'] = {
        'scale': qwen_gemma_final_norm_scale
    }
    def get_params_for_layer(current_layer):
        layer_s_inputlayernorm_torch_tensor = qwen_hf_model.model.layers[layer_s].input_layernorm._parameters['weight'].data 
        layer_s_inputlayernorm_jax_tensor = t2j(layer_s_inputlayernorm_torch_tensor.cpu().detach()) #- 1. # subtract 1. because gemma codebase uses rmsnormweights = 1 + scale
        # layer_0.attn.q_einsum
        #layer_s_qproj_torch_tensor = current_layer.self_attn.q_proj.weight.data
        #layer_s_qeinsum_jax = t2j(layer_s_qproj_torch_tensor.reshape(qwen_model_config['num_attention_heads'], qwen_model_config['hidden_size'], -1))
        # layer_0.attn.kv_einsum
        layer_s_kv_matrix_torch_tensor = torch.stack((current_layer.self_attn.k_proj.weight.data, 
                current_layer.self_attn.v_proj.weight.data))
        kv_einsum = t2j(layer_s_kv_matrix_torch_tensor)
        kv_einsum = kv_einsum.reshape(2, qwen_model_config['num_key_value_heads'], qwen_model_config['hidden_size'],  qwen_model_config['head_dim'])
        # layer_0.attn.attn_vec_einsum
        layer_s_o_proj = t2j(current_layer.self_attn.o_proj.weight.data)
        layer_s_attn_vec_einsum = layer_s_o_proj.reshape(qwen_model_config['num_attention_heads'], qwen_model_config['head_dim'], qwen_model_config['hidden_size'])
        # layer_0.mlp.gating_einsum
        layer_s_mlp_gating_einsum = t2j(torch.stack((current_layer.mlp.gate_proj.weight.data,
                    current_layer.mlp.up_proj.weight.data)))
        # layer_0.mlp.linear
        layer_s_mlp_linear = t2j(current_layer.mlp.down_proj.weight.data.T)
        # layer_0.post_attention_norm.scale
        layer_s_post_attention_layernorm = t2j(current_layer.post_attention_layernorm.weight.data)
        # layer_0.post_ffw_norm
        layer_s_pre_ffw_norm = None
        layer_s_post_ffw_norm = None
        # layer_0.attn._query_norm
        layer_s_attn_query_norm = t2j(qwen_hf_model.model.layers[0].self_attn.q_norm.weight.data)
        layer_s_params = {
            'attn': {
                '_key_norm': None, # qwen does not use it
                '_query_norm': {
                    'scale':  layer_s_attn_query_norm
                },
                'attn_vec_einsum': {
                    'w': layer_s_attn_vec_einsum
                },
                'q_einsum': {
                    'w': get_q_einsum_for_layer(current_layer) #layer_s_qeinsum_jax
                },
                'kv_einsum': {
                    'w': kv_einsum
                },
            },
            'mlp': {
                'gating_einsum': layer_s_mlp_gating_einsum,
                'linear': layer_s_mlp_linear,
            },
            'post_attention_norm': {
                'scale': t2j(current_layer.post_attention_layernorm.weight.data)
            },
            'post_ffw_norm': None,
            'pre_attention_norm': {
                'scale': layer_s_inputlayernorm_jax_tensor
            },
            'pre_ffw_norm': {
                'scale': layer_s_inputlayernorm_jax_tensor
            }
        }
        return layer_s_params
    for s in range(qwen_model_config['num_hidden_layers']):
        layer_s = 0
        current_layer = qwen_hf_model.model.layers[layer_s]
        layer_s_params = get_params_for_layer(current_layer)
        qwengemma_params[f'layer_{s}'] = layer_s_params  
    return qwengemma_params

#my_qwengemma_params = get_qwengemma06b_params()
#print(my_qwengemma_params.keys())