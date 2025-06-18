# qwemma

Jax is nice, and smart people at Google are maintaining the public Gemma codebase.

With qwemma I want to make Qwen3 work with the Gemma codebase.

## Part 1
In Part1, using HuggingFace Transformers, I download Qwen3-0.6B and get the probability distribution for next-token for the string "hey there, how are"

## Part 2
In Part 2, using Gemma code base, I download Gemma3-1b and get the probability distirbution for next-token for the string "hey there, how are"

## Part 3
Gemma code base uses a dictionary for the parameters for the LLM. In Part 3, I load Huggingface Qwen3-0.6B and build a dictionary for making Qwen3 work with the Gemma codebase. I am not yet concerned with all the details.

## Part 4
In Part 4, I define the preliminary Qwen3Gemma3-06B model similarly to how the Gemma models are defined in the Gemma codebase. I set the hyperparameters using the Qwen3-0.6B config.json from Huggingface. I show a preliminary forward pass, that is then ready to be adapted to make sure the computation agrees with Huggingface Qwen3.

## Part 5
In Part 5, I want to ensure that the embedding matrices and residual streams agrees up to the point where we start applying the first Transformer block.