# Part 4 - the purpose of this script is to use the
# qwengemmaparams dict with the gemma codebase and get to
# a point where there is no error when using it to
# predict the next token for "hi there how are".
# we do not require yet that qwengemma corresponds to
# gwen, just that it uses as much as possible of the
# qwemgemmaparamsdict as possible, so we also have the
# sizes right etc in the hyperparams class.
# %%
import gemma
import jax
from gemma import gm
from etils import epath 
import jax.numpy as jnp
from gemma import transformer
from gemma.gm.nn import _transformer
from gemma.gm.ckpts import _paths
from gemma.transformer import _NUM_LAYERS_GEMMA3_1B, make_attention_layers_types, GEMMA3_ATTENTION_PATTERN, QueryPreAttentionNormalisation
from transformers import AutoTokenizer, AutoModelForCausalLM
import treescope
# %%
from part3_qwen_params_dict import get_qwengemma06b_params, qwen_model_config, t2j
# %%

class Qwen3Gemma3_06BConfig(transformer.TransformerConfig):
  """Custom configuration for Qwen3Gemma3_06B."""

  @classmethod
  def qwen3gemma3_06b(cls):
    return cls(
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
        use_qk_norm=False,
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_1B
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=512,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        vision_encoder=None,
        qwemma_setting_rescale_embeddings=False
    )

class Qwen3Gemma3_06B(_transformer.Transformer):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture."""

  # Correct the method call to match the method name in Qwen3Gemma3_06BConfig
  config: transformer.TransformerConfig = (
    Qwen3Gemma3_06BConfig.qwen3gemma3_06b()
  )

  INFO = _transformer.ModelInfo(
      tokenizer_version=3,
      default_ckpt=_paths.CheckpointPath.GEMMA3_1B_IT,
  )


def get_gemma_params():
    gemma_local_params_epath_str = './params/gemma3_1b_it' # This was for Orbax
    gemma_local_params_absolute_epath = epath.Path(gemma_local_params_epath_str).resolve() # Keep for cleanup if needed
    gemma_params = gm.ckpts.load_params(gemma_local_params_absolute_epath)
    return gemma_params

def get_gemma_tokenized(input_strings):
    gemma_tokenizer = gm.text.Gemma3Tokenizer()
    gemmaprompt = gemma_tokenizer.encode(input_strings, add_bos=True)
    gemmaprompt_jnparray = jnp.asarray(gemmaprompt)
    return gemmaprompt_jnparray

def get_qwemma_tokenized(input_strings):
    qwen_hf_model_name = "Qwen/Qwen3-0.6B"
    qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
    qwen_hf_model_inputs = qwen_hf_tokenizer(input_strings, return_tensors="pt", padding=True)
    qwen_hf_model_inputs
    qwenprompt_jnparray = t2j(qwen_hf_model_inputs.input_ids)
    return qwenprompt_jnparray

def do_jax_forward_pass(input_ids, model, params_dict):
    out = model.apply(
        {'params': params_dict},
        tokens=input_ids,
        return_last_only=True,  # Only predict the last token
        capture_intermediates=True
    )
    return out


model_gemma = gm.nn.Gemma3_1B()
model_qwemma = Qwen3Gemma3_06B()
gemma_params = get_gemma_params()
qwemma_params = get_qwengemma06b_params()
prompt = "hi there how are"
#gemmaprompt_input_ids = get_gemma_tokenized(prompt)
qwemmaprompt_input_ids = get_qwemma_tokenized(prompt)

#print('gemma forward pass')
#out_gemma = do_jax_forward_pass(gemmaprompt_input_ids, model_gemma, gemma_params)

#print('qwemma forward pass')
#out_qwemma = do_jax_forward_pass(qwemmaprompt_input_ids, model_qwemma, qwemma_params)