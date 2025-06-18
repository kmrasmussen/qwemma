# Purpose of this script is just to check that we can download Qwen-0.6B from HuggingFace Transformers
# and get next-token probability distribution for "hi, there ..."

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
2+2
# %%
model_name = "Qwen/Qwen3-0.6B"
# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
# %%
model
# %%
model.forward()
# %%
# systemprompt yo, you are a chatbot, user: hi there" assistant:
messages = [
    {"role": "user", "content": "hi there"}
]

# %%
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
# %%
text
# %%
model_inputs = tokenizer([text, "hi there, how are"], return_tensors="pt", padding=True)
# %%
model_inputs
# %%
outputs = model(**model_inputs)
# %%
outputs
# %%
outputs.logits.shape
# %%
hithere_logits = outputs.logits[0,-1] # [1,4]
# %%
hithere_logits

import torch
# %%

# %%
tokenizer.decode([torch.argmax(hithere_logits)])
# %%
