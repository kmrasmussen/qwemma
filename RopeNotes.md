# Rotational embedddings

We will look at Rotational embeddings from a concrete perspective of how it is implemented in Huggingface Transformers for Qwen3.

The whole Qwen3 model has a module called Qwen3RotaryEmbedding. This computes some stuff that can be reused within the different attention layers.

First, the model config describes a hyper parameter called rope $\theta$.

In the attention there is also a head dim $H$, the dimensionality of the key and query vectors. 

With these two it is possible to compute $H$ different "inverse frequencies" like this
```
inv_freq_h = 1.0 / (rope_theta ^ (2 * h / H))
```