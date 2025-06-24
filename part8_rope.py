# Part 8 - the goal of this script is to make the Gemma Rope implementation perform transformation that is equivalent
# to the Huggingface Qwen Rope transformation, with as little modifications to the Gemma codebases as possible.

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
from gemma import positional_embeddings

# %%
qwemma_params = get_qwengemma06b_params()
# %%
qwen_hf_model_name = "Qwen/Qwen3-0.6B"
qwen_hf_tokenizer = AutoTokenizer.from_pretrained(qwen_hf_model_name)
qwen_hf_model_inputs = qwen_hf_tokenizer(["hi there how are"], return_tensors="pt", padding=True)
qwen_hf_model = AutoModelForCausalLM.from_pretrained(qwen_hf_model_name, torch_dtype=torch.bfloat16)
# %%
qwen_hf_model.model.rotary_emb
# %%
b = 3
d = qwen_model_config['hidden_size']
t = 4
n, h = 8, 128
key = jax.random.key(0)
key_x, key = jax.random.split(key)
x = jax.random.normal(key_x, (b,t,n,h)).astype(jnp.bfloat16)
segment_pos = jnp.tile(jnp.arange(t), (b,1)) #jnp
# %%
x_pt = j2t_bfloat16(x)

# %%
x_pt.shape
# %%
segment_pos
# %%
x_pt.shape
segment_pos_pt = j2t_bfloat16(segment_pos)
# %%
hf_qwen_cos, hf_qwen_sin = qwen_hf_model.model.rotary_emb(x_pt, segment_pos_pt)
# %%
hf_qwen_cos.shape
# %%
# %%
qwen_model_config
# %%
qwemma_rope_output, qwemma_cos, qwemma_sin, qwemma_fractions, qwemma_timescale, qwemma_sinusoid_inp = positional_embeddings.apply_rope(
        x,
        segment_pos,
        base_frequency=1000000,
        scale_factor=1.0,
    )
# %%
qwemma_sinusoid_inp.shape
# %%
qwemma_fractions.shape
# %%
x.shape
# %%
qwemma_rope_output.shape
# %%
qwemma_cos.shape
# %%
hf_qwen_cos.shape
# %%
qwen_rotary = qwen_hf_model.model.rotary_emb 
# %%
qwen_inv_freq, qwen_att_scaling = qwen_rotary.rope_init_fn(qwen_rotary.config, "cpu")
# %%
qwen_att_scaling
# %%
qwen_model_config
# %%
qwen_inv_freq[-1]
# %%
1.0 / (1000000 ** (2. * 63 / 128))
# %%
class HfQwen3RotaryEmbedding(torch.nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = qwen_rotary.rope_init_fn

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            print('hf rotary embd', self.attention_scaling)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype), freqs

# %%
qwen_rotary2 =  HfQwen3RotaryEmbedding(qwen_rotary.config)
# %%
hf_cos, hf_sin, freqs = qwen_rotary2(x_pt, segment_pos_pt)
# %%
freqs.shape
# %%
qwen_model_config
# %%
qwemma_sinusoid_inp.shape
# %%
freqs[-1,-1,-1]
# %%
qwemma_sinusoid_inp[-1,-1,-1,-1]
# %%
freqs.shape
# %%
qwemma_sinusoid_inp
# %%
hf_sinusoid_inp = t2j(freqs)[:, :, None, :]
# %%
jnp.allclose(qwemma_sinusoid_inp, hf_sinusoid_inp)
# %%
jnp.max(qwemma_sinusoid_inp - hf_sinusoid_inp)
# %%
qwemma_cos.shape
# %%
hf_cos.shape
# %%

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
# %%
x_pt.shape
# %%
hf_cos.shape
# %%
hf_cos_expanded = hf_cos.unsqueeze(2)  # Shape becomes (3, 4, 1, 128)
hf_sin_expanded = hf_sin.unsqueeze(2) 
# %%
hf_q_embed, _ = hf_apply_rotary_pos_emb(x_pt, x_pt, hf_cos, hf_sin, unsqueeze_dim=2)
# %%
hf_q_embed.shape
# %%
qwemma_rope_output.shape
# %%
hf_q_embed_j = t2j(hf_q_embed)
# %%
qwemma_rope_output.dtype
# %%
jnp.max(qwemma_rope_output - hf_q_embed_j)
# %%
