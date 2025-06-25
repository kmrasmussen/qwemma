# %%
import torch, jax, jax.numpy as jnp
import numpy as np

# --- Make the *same* data in both frameworks ------------------

key = jax.random.key(0)
x_np = jax.random.normal(key,  (4, 10, 64), dtype=jnp.float32)  # (B,T,D)
w_np = jax.random.normal(key,  (64, 32),    dtype=jnp.float32)  # (D,K)

# PyTorch copies
x_pt = torch.tensor(np.array(x_np), device='cuda', dtype=torch.float32)
w_pt = torch.tensor(np.array(w_np), device='cuda', dtype=torch.float32)

# JAX copies
x_jx = x_np            # already on device (GPU/TPU) if JAX is so configured
w_jx = w_np

# --- Flags for fair comparison --------------------------------

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32  = False
torch.use_deterministic_algorithms(True)

from jax import config
config.update("jax_default_matmul_precision", "bfloat16")  # or "float32"
# (enable_x64 only matters if you want float64 later)

# --- FP32 baseline --------------------------------------------

out_pt = torch.einsum('btd,dk->btk'_
import torch, jax, jax.numpy as jnp
import numpy as np

# --- Make the *same* data in both frameworks ------------------

key = jax.random.PRNGKey(0)
x_np = jax.random.normal(key,  (4, 10, 64), dtype=jnp.float32)  # (B,T,D)
w_np = jax.random.normal(key,  (64, 32),    dtype=jnp.float32)  # (D,K)

# PyTorch copies
x_pt = torch.tensor(np.array(x_np), device='cuda', dtype=torch.float32)
w_pt = torch.tensor(np.array(w_np), device='cuda', dtype=torch.float32)

# JAX copies
x_jx = x_np            # already on device (GPU/TPU) if JAX is so configured
w_jx = w_np

# --- Flags for fair comparison --------------------------------

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32  = False
torch.use_deterministic_algorithms(True)

from jax import config
config.update("jax_default_matmul_precision", "bfloat16")  # or "float32"
# (enable_x64 only matters if you want float64 later)

# --- FP32 baseline --------------------------------------------

out_pt = torch.einsum('btd,dk->btk', x_pt.float(), w_pt.float())        # PyTorch
out_jx = jnp.einsum('btd,dk->btk', x_jx.astype(jnp.float32),
                                  w_jx.astype(jnp.float32))            # JAX

diff_fp32 = np.max(np.abs(out_pt.cpu().numpy() - np.array(out_jx)))
print("max abs diff (FP32):", diff_fp32)

# --- BF16 run --------------------------------------------------

x_pt16 = x_pt.to(torch.bfloat16); w_pt16 = w_pt.to(torch.bfloat16)
x_jx16 = x_jx.astype(jnp.bfloat16); w_jx16 = w_jx.astype(jnp.bfloat16)

out_pt16 = torch.einsum('btd,dk->btk', x_pt16, w_pt16)
out_jx16 = jnp.einsum('btd,dk->btk', x_jx16, w_jx16)

diff_bf16 = np.max(np.abs(out_pt16.float().cpu().numpy() -
                          out_jx16.astype(jnp.float32)))
print("max abs diff (BF16):", diff_bf16)

# %%
