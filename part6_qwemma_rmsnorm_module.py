# Part 6 - The goal of this script is to ensure that the QwemmaRMSNorm and the Huggingface Transformers Qwen3RMSNorm modules behave the same.
# %%
from flax import linen
import jax
import jax.numpy as jnp
import torch
import numpy as np

def j2t_bfloat16(x_jax: jnp.ndarray) -> torch.Tensor:
    """JAX → PyTorch, keeps bfloat16 and forces default (row-major) layout."""
    return torch.utils.dlpack.from_dlpack(x_jax.__dlpack__()).contiguous()

def t2j(x_torch: torch.Tensor) -> jnp.ndarray:
    """PyTorch → JAX, robust to any odd strides."""
    return jax.dlpack.from_dlpack(x_torch.detach().contiguous(), copy=True)
# %%
class QwemmaRMSNorm(linen.Module):
  """RMSNorm layer."""

  @linen.compact
  def __call__(self, x):
    scale = self.param('scale', linen.initializers.zeros_init(), (x.shape[-1]))
    x_float32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float32), axis=-1, keepdims=True)
    rescaled_x = x_float32 * jax.lax.rsqrt(variance + 1e-6)
    rescaled_x_bfloat16 = rescaled_x.astype(jnp.bfloat16)
    normed_inputs = rescaled_x_bfloat16 * scale
    return normed_inputs

class HfQwen3RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        ret = self.weight * hidden_states.to(input_dtype)
        return ret

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
# %%
rmsnorm = QwemmaRMSNorm()
key = jax.random.key(0)
key_x, key = jax.random.split(key)
key_scale, key = jax.random.split(key)
d = 10
b = 1
t = 4
x = jax.random.normal(key_x,(b,t,d),dtype=jnp.bfloat16)
scale = jax.random.normal(key_scale,(d),dtype=jnp.bfloat16)
params = {'scale': scale}
qwemma_rms_out = rmsnorm.apply({'params': params}, x=x)
scale_pt = j2t_bfloat16(scale)
scale_pt_then_j = t2j(scale_pt)
scale, scale_pt, scale_pt_then_j
assert jnp.allclose(scale,scale_pt_then_j), "Torch and Jax Scale weights do not agree"
x_pt = j2t_bfloat16(x)
x.shape, x_pt.shape
x_pt_then_j = t2j(x_pt)
myhfqwenrmsnorm = HfQwen3RMSNorm(d)
myhfqwenrmsnorm.weight.data = scale_pt
hf_rmsnorm_output = myhfqwenrmsnorm.forward(x_pt)
hf_rmsnorm_output_j = t2j(hf_rmsnorm_output)
assert jnp.allclose(hf_rmsnorm_output_j, qwemma_rms_out), "RMSNorm outputs do not agree"
# %%
jnp.allclose(hf_rmsnorm_output_j, qwemma_rms_out)
# %%
