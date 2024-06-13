# Mamba implementation on the induction head task 🐍
Adapted from the implementation of [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py) and [hrbigelow/mamba-reall] (https://github.com/hrbigelow/mamba-recall) in PyTorch with architectural modifications. This is the plaintext, HE-friendly version of a two-layer Mamba for secure inference.

## Updates
- <b>13/06/2024</b> : Train a Mamba model on the induction head task 
___
## Overview

<u>The repo is organized as follows : </u>
- `pscan.py` : a PyTorch implementation of Blelloch's parallel scan
- `mamba.py` : the Mamba model, as described in the [paper](https://arxiv.org/abs/2312.00752). It is numerically equivalent (initialization, forward and backward pass).
- `mamba_lm.py` : encapsulates a Mamba model in order to use it as a language model
- `main_induction_head.py` : trains and evaluates a Mamba model and on the induction head task
- `📁 docs` : a folder containing annotated explanations about the code, focusing on the parallel scan

___
## Sources and where to learn more
- the [Mamba paper](https://arxiv.org/abs/2312.00752) : describes the Mamba architecture as implemented in this repo, which allows to model sequences in linear time.
- the [Mamba implementation](https://github.com/state-spaces/mamba), which is written in PyTorch but uses a parallel scan written in CUDA. This is the version that is the fastest. 
- [mamba.py 🐍 : a simple and efficient Mamba implementation](https://github.com/alxndrTL/mamba.py) : a straightforward implementation of Mamba.
- [Experiments with Mamba State Space Models](https://github.com/hrbigelow/mamba-recall): experiments Mamba on the synthetic data induction heads task


## TODOs
- Retrain a model of a smaller scale by reducing model dimensions and number of layers.
- Approximate activation functions (SiLU, Softplus) with polynomials
- Study the effect of removing RMSNorm