# Purpose of this script is to check that we can get
# Gemma 1b to predict next token on some string like "hi there"

# %%
import gemma
from gemma import gm
from etils import epath 
import jax.numpy as jnp
# %%
gemma.__file__
model = gm.nn.Gemma3_1B()
# %%
local_params_epath_str = './params/gemma3_1b_it' # This was for Orbax
local_params_absolute_epath = epath.Path(local_params_epath_str).resolve() # Keep for cleanup if needed
# %%
params = gm.ckpts.load_params(local_params_absolute_epath )
# %%
tokenizer = gm.text.Gemma3Tokenizer()
# %%
print('tokenizing')
prompt = tokenizer.encode('hi there, how are', add_bos=True)
prompt = jnp.asarray(prompt)
# %%
print('prompt', prompt)
print('forward pass')
# %%
out = model.apply(
    {'params': params},
    tokens=prompt,
    return_last_only=True,  # Only predict the last token
)
print('out', out)
# %%
out
# %%
out.logits.shape
# %%
jnp.argmax(out.logits)
# %%
tokenizer.decode(611)
# %%
