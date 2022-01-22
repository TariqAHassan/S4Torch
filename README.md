# S4 Torch

A PyTorch implementation of [Structured State Space for Sequence Modeling (S4)](https://arxiv.org/abs/2111.00396), 
based on the beautiful [Annotated S4](https://srush.github.io/annotated-s4/) blog post
by [@srush](https://github.com/srush) and [@siddk](https://github.com/siddk).

## Installation

```sh
pip install git+git://github.com/TariqAHassan/s4torch@main
```

Requires Python 3.8+.

## Quick Start

The `S4Model()` provides a high-level implementation of the S4 model.
A simple example is provided below.

```python
import torch
from s4torch import S4Model

N = 32
d_input = 1
d_model = 128
d_output = 128
n_blocks = 3
seq_len = 784

u = torch.randn(1, seq_len, d_input)

s4model = S4Model(
    d_input,
    d_model=d_model,
    d_output=d_output,
    n_blocks=n_blocks,
    n=N,
    l_max=seq_len,
    collapse=False,  # if `True` average predictions over time
)
assert s4model(u).shape == (*u.shape[:-1], s4model.d_output)
```

## Components

### Layer

The `S4Layer()` implements the core logic of S4.

```python
import torch
from s4torch.layer import S4Layer

N = 32
d_model = 128
seq_len = 784

u = torch.randn(1, seq_len, d_model)

s4layer = S4Layer(d_model, n=N, l_max=seq_len)
assert s4layer(u).shape == u.shape
```

### Block

The `S4Block()` embeds `S4Layer()` in a commonplace processing "pipeline",
with a `GELU()` activation, dropout, linear layer, skip connection and layer normalization.
(`S4Model()`, above, is composed of these blocks.)

```python
import torch
from s4torch.block import S4Block

N = 32
d_input = 1
d_model = 128
d_output = 128
seq_len = 784

u = torch.randn(1, seq_len, d_model)

s4block = S4Block(d_model, n=N, l_max=seq_len)
assert s4block(u).shape == u.shape
```

## References

The S4 model was developed by Albert Gu, Karan Goel, and Christopher Ré. 
If you find this repository useful, please cite their (extremely impressive) paper:

```bibtex
@misc{gu2021efficiently,
    title={Efficiently Modeling Long Sequences with Structured State Spaces}, 
    author={Albert Gu and Karan Goel and Christopher Ré},
    year={2021},
    eprint={2111.00396},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Also consider checking out their fanstic repository at [github.com/HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces).
