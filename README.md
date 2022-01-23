# S4 Torch

A PyTorch implementation of [Structured State Space for Sequence Modeling (S4)](https://arxiv.org/abs/2111.00396), 
based on the beautiful Annotated S4 [blog post](https://srush.github.io/annotated-s4/)
and JAX-based [library](https://github.com/srush/annotated-s4/) by [@srush](https://github.com/srush) and 
[@siddk](https://github.com/siddk).

## Installation

```sh
pip install git+git://github.com/TariqAHassan/s4torch
```

Requires Python 3.9+.

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
    collapse=False,  # if `True` average predictions over time prior to decoding
)
assert s4model(u).shape == (*u.shape[:-1], s4model.d_output)
```

## Training

Models can be trained using the command line interface (CLI) provided by `train.py`.

**Notes**:
 * development requirements must be installed prior to training. This can be accomplished by 
   running `pip install -r dev_requirements.txt`.
 * average pooling is used in some training sessions described below. The primary motivation  
   for the use of pooling is to reduce memory usage, and it is not used in the original implementation.
   Pooling can be disabled by setting `--pooling=None`, or by simply omitting the `--pooling` flag.
 * all experiments were performed on a single NVIDIA® Tesla® V100 GPU with 16 GB of vRAM

#### [MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST)

```sh
python train.py --dataset=mnist --batch_size=16
```

**Validation Accuracy**: 98.6% after 4 epochs <br>
**Speed**: ~11.5 batches/second

#### [CIFAR10](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10)

```shell
python train.py \
  --dataset=CIFAR10 \
  --batch_size=32 \
  --max_epochs=200 \
  --lr=1e-2 \
  --n_blocks=6 \
  --pooling=avg_2 \
  --d_model=1024 \
  --weight_decay=0.01 \
  --p_dropout=0.25 \
  --patience=20
```

**Validation Accuracy**: 77.0% after 9 epochs <br>
**Speed**: ~1.6 batches/second

#### [SpeechCommands](https://pytorch.org/audio/stable/datasets.html#torchaudio.datasets.SPEECHCOMMANDS)

```sh
python train.py \
  --dataset=speechcommands10 \
  --batch_size=16 \
  --max_epochs=150 \
  --lr=1e-2 \
  --n_blocks=6 \
  --pooling=avg_2 \
  --d_model=128 \
  --weight_decay=0.0 \
  --norm_type=batch,
  --p_dropout=0.1 \
  --patience=10
```

**Validation Accuracy**: TBD <br>
**Speed**: TBD

Note that the `speechcommands10` dataset uses a subset of 10 speech commands, as 
in the [original implementation](https://github.com/HazyResearch/state-spaces#speech-commands) of S4.
If you would like to train against all speech commands, you can use the `speechcommands` dataset instead.

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
with an activation function, dropout, linear layer, skip connection and layer normalization.
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
If you find the S4 model useful, please cite their impressive paper:

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

Also consider checking out their fantastic repository at [github.com/HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces).
