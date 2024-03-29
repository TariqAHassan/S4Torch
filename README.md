# S4 Torch

A PyTorch implementation of [Structured State Space for Sequence Modeling (S4)](https://arxiv.org/abs/2111.00396), 
based on the beautiful Annotated S4 [blog post](https://srush.github.io/annotated-s4/)
and JAX-based [library](https://github.com/srush/annotated-s4/) by [@srush](https://github.com/srush) and 
[@siddk](https://github.com/siddk).

## Installation

```sh
pip install git+https://github.com/TariqAHassan/s4torch
```

If you wish to perform development and/or use wavelet-based transforms, you
will need to install the development requirements. This can be done with:

```shell
pip install -r dev_requirements.txt
```

Requires Python 3.9+.

## Quick Start

The `S4Model()` provides a high-level implementation of the S4 model, as illustrated below.

```python
import torch
from s4torch import S4Model

N = 32
d_input = 1
d_model = 128
n_classes = 10
n_blocks = 3
seq_len = 784

u = torch.randn(1, seq_len, d_input)

s4model = S4Model(
    d_input,
    d_model=d_model,
    d_output=n_classes,
    n_blocks=n_blocks,
    n=N,
    l_max=seq_len,
    collapse=True,  # average predictions over time prior to decoding
)
assert s4model(u).shape == (u.shape[0], n_classes)
```

## Training

Models can be trained using the command line interface (CLI) provided by `train.py`. <br>
CLI documentation can be obtained by running `python train.py --help`.

**Notes**:
 * development requirements must be installed prior to training. This can be accomplished by 
   running `pip install -r dev_requirements.txt`.
 * average pooling after each S4 block is used in some training sessions described below, whereas the 
   original S4 implementation only uses average pooling prior to decoding. The primary motivation for 
   additional pooling was to reduce memory usage and, at least in the case of Sequential MNIST, does not appear 
   reduce accuracy. These additional pooling layers can be disabled by setting `--pooling=None`, or by simply 
   omitting the `--pooling` flag.
 * specifying `--batch_size=-1` will result in the batch size being 
   [auto-scaled](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#batch-size-finder)
 * all experiments were performed on a machine with 8 CPU cores, 30 GB of RAM and a single 
   NVIDIA® Tesla® V100 GPU with 16 GB of vRAM

#### Sequential [MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST)

```sh
python train.py \
  --dataset=smnist \
  --batch_size=16 \
  --max_epochs=100 \
  --lr=1e-2 \
  --n_blocks=6 \
  --d_model=128 \
  --norm_type=layer
```

**Validation Accuracy**: 98.6% after 5 epochs, 99.3% after 9 epochs (best) <br>
**Speed**: ~10.5 batches/second

```sh
python train.py \
  --dataset=smnist \
  --batch_size=16 \
  --pooling=avg_2 \
  --max_epochs=100 \
  --lr=1e-2 \
  --n_blocks=6 \
  --d_model=128 \
  --norm_type=layer
```

**Validation Accuracy**: 98.4% after 5 epochs, 99.3% after 10 epochs (best) <br>
**Speed**: ~11.5 batches/second

#### Permuted [MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST)

```sh
python train.py \
  --dataset=pmnist \
  --batch_size=16 \
  --pooling=avg_2 \
  --max_epochs=100 \
  --lr=1e-2 \
  --n_blocks=6 \
  --d_model=128 \
  --norm_type=layer
```

**Validation Accuracy**: 94.0% after 5 epochs, 96.2% after 18 epochs (best) <br>
**Speed**: ~11.5 batches/second

#### Sequential [CIFAR10](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CIFAR10)

```shell
python train.py \
  --dataset=scifar10 \
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

**Validation Accuracy**: 75.0% after 8 epochs, 79.3% after 15 epochs (best) <br>
**Speed**: ~1.6 batches/second

#### [SpeechCommands](https://pytorch.org/audio/stable/datasets.html#torchaudio.datasets.SPEECHCOMMANDS)

```sh
python train.py \
  --dataset=speech_commands10 \
  --batch_size=-1 \
  --max_epochs=150 \
  --lr=1e-2 \
  --n_blocks=6 \
  --pooling=avg_2 \
  --d_model=128 \
  --weight_decay=0.0 \
  --norm_type=batch \
  --norm_strategy=post \
  --p_dropout=0.1 \
  --patience=10
```

**Validation Accuracy**: 93.2% after 5 epochs, 95.8% after 13 epochs (best) <br>
**Speed**: ~2.1 batches/second

Notes:
  
  * the `speech_commands10` dataset uses a subset of 10 speech commands, as in the 
    [original implementation](https://github.com/HazyResearch/state-spaces#speech-commands) of S4. 
    If you would like to train against all speech commands, the `speech_commands` dataset can be used instead.
  * Batch normalization appears to work best with a "post" normalization strategy, whereas 
    a "pre" normalization strategy appears to work best with layer normalization.

#### [NSynth](https://magenta.tensorflow.org/datasets/nsynth)

##### Raw Waveform

```sh
python train.py \
  --dataset=nsynth_short \
  --batch_size=-1 \
  --val_prop=0.01 \
  --max_epochs=150 \
  --limit_train_batches=0.025 \
  --lr=1e-2 \
  --n_blocks=4 \
  --pooling=avg_2 \
  --d_model=128 \
  --weight_decay=0.0 \
  --norm_type=batch \
  --norm_strategy=post \
  --p_dropout=0.1 \
  --precision=16 \
  --accumulate_grad=4 \
  --patience=10
```

**Validation Accuracy**: 39.6% after 5 epochs, 54.1% after 17 epochs (best) <br>
**Speed**: ~1.6 batches/second

Notes:

  * The model is tasked with classifying waveforms based on the musical instrument which generated them (10 classes)
  * The `nsynth_short` dataset contains waveforms which are truncated after 2 seconds, whereas the `nsynth` dataset contains 
    the full four-second waveforms.

##### Continuous Wavelet Transform (`|CWT(x)|`)

```sh
python train.py \
  --dataset=nsynth_short \
  --batch_size=-1 \
  --val_prop=0.01 \
  --max_epochs=150 \
  --limit_train_batches=0.025 \
  --lr=1e-2 \
  --n_blocks=6 \
  --pooling=avg_2 \
  --d_model=100 \
  --weight_decay=0.0 \
  --norm_type=batch \
  --norm_strategy=post \
  --p_dropout=0.1 \
  --precision=16 \
  --accumulate_grad=1 \
  --wavelet_tform=True \
  --patience=10
```

**Validation Accuracy**: 52.7% after 5 epochs, 69.4% after 72 epochs (best) <br>
**Speed**: ~1.3 batches/second

Notes:

   * This experiment uses the magnitude of the CWT (with a morlet wavelet) as the input
     representation. This produces a (rather substantial) 15%+ increase in performance.
   * **This requires that you have [`pycwt`](https://github.com/regeirk/pycwt) installed. See the [Installation](#installation) instructions above.** 

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
    author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
    year={2021},
    eprint={2111.00396},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Also consider checking out their fantastic repository at [github.com/HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces).
