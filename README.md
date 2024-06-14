# Mamba implementation on the induction head task 🐍
Adapted from the implementation of [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py) and [hrbigelow/mamba-reall] (https://github.com/hrbigelow/mamba-recall) in PyTorch with architectural modifications. This is the plaintext, HE-friendly version of Mamba for secure inference. The model consists of two MambaBlock layers with polynomially approximated SiLU and Softplus activation functions. RMSNorm is only applied once before the output layer that converts embeddings back to tokens and thus does not need to be included in the HE implementation. Plaintext sequences in different lengths (e.g 128, 256, 512, ...) are first transformed into embeddings on the clien side, and then get encrypted for secure inference; the results, after decryption, are the output logits from the model and needs to be normalized and mapped back to plaintext tokens.

## Updates
- <b>14/06/2024</b> : Train a HE-friendly Mamba model on the induction head task 

## Overview
<u>The repo is organized as follows : </u>
- `pscan.py` : a PyTorch implementation of Blelloch's parallel scan
- `mamba.py` : the Mamba model, as described in the [paper](https://arxiv.org/abs/2312.00752). It is numerically equivalent (initialization, forward and backward pass).
- `mamba_lm.py` : encapsulates a Mamba model in order to use it as a language model
- `main_induction_head.py` : trains and evaluates a Mamba model and on the induction head task. The pretrained model will be stored under the "saves" folder once the training completes.
- `📁 docs` : a folder containing annotated explanations about the code, focusing on the parallel scan

## The induction head task
The induction head task evaluates a model's ability to copy and complete sequences that have occurred before and is often tied to the in-context learning of large models. Given an anchor token, the task is to perform associative recall and copy to predict the token in its next immediate position. 
For example, if the model has seen the bigram "Harry Potter" in the sequence, then the next time "Harry" appears in the same sequence, the model should be able to predict "Potter" by copying from history.
```markdown
Input: tensor([[ 8, 12,  8,  6, 10, 11,  2,  5, **16**,  **6**,  0,  4,  1,  3, 10, 14,  5,  8,
         14,  0,  1, 12, 13, 15,  9, 14,  3, 10, 14, 12,  0,  7, 10,  2, 10, 15,
         10, 15,  2,  7,  3,  9, 12,  1,  2, 15,  4,  7,  5,  1, 11,  1,  5,  6,
         15, 10,  1, 14,  4,  2,  8,  2,  3,  0,  6,  9,  0,  3,  4,  2,  3,  8,
          6,  1,  0, 10, 14, 15,  0,  7, 12,  2,  6,  3,  2,  0,  4,  2,  6,  0,
          9,  5,  8,  4,  7, 10, 13, 10,  7, 15,  3,  7,  3,  5,  9, 10, 14,  0,
          3,  9, 10,  9,  3, 14,  3,  8,  6, 13,  9,  4,  9,  2, 10,  6, 10,  4,
         13,  1, 16]], device='cuda:0')
Predicted output: tensor([6], device='cuda:0')
```

___
## Sources and where to learn more
- the [Mamba paper](https://arxiv.org/abs/2312.00752) : describes the Mamba architecture as implemented in this repo, which allows to model sequences in linear time.
- the [Mamba implementation](https://github.com/state-spaces/mamba), which is written in PyTorch but uses a parallel scan written in CUDA. This is the version that is the fastest. 
- [mamba.py 🐍 : a simple and efficient Mamba implementation](https://github.com/alxndrTL/mamba.py) : a straightforward implementation of Mamba.
- [Experiments with Mamba State Space Models](https://github.com/hrbigelow/mamba-recall): experiments Mamba on the synthetic data induction heads task
