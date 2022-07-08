# PyCWT is released under a BSD-style open source licence:
#
# Copyright (c) 2017 Sebastian Krieger, Nabil Freij, Alexey Brazhe,
# Christopher Torrence, Gilbert P. Compo and contributors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

    Continuous Wavelet Transform

    References:
        * Implementation based on https://github.com/regeirk/pycwt

"""
from __future__ import annotations

import numpy as np
import torch
from pycwt.wavelet import _check_parameter_wavelet
from torch import nn

from s4torch.dsp.utils import is_pow2, next_pow2, pow2pad


class Cwt(nn.Module):
    def __init__(
        self,
        n0: int,  # duration
        dt: float = 1.0,
        dj: float = 1 / 12,
        s0: int = -1,
        J: int = -1,
        wavelet: str = "morlet",
        freqs: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if not is_pow2(n0):
            raise ValueError(f"n0 is not a power of 2, got {n0}")
        self.n0 = n0
        self.dt = dt
        self.dj = dj
        self.s0 = s0
        self.J = J
        self.wavelet = _check_parameter_wavelet(wavelet)
        self.freqs = freqs

        # This tensor is too large store as a buffer on GPU.
        # However, performance is still improved by caching
        # it on CPU memory and transferring to GPU as needed.
        self.psi_ft_bar = self._get_psi_ft_bar()

    def _get_params(self) -> tuple[int, int, int]:
        if self.freqs is None:
            # Smallest resolvable scale
            if self.s0 == -1:
                s0 = 2 * self.dt / self.wavelet.flambda()
            else:
                s0 = self.s0
            # Number of scales
            if self.J == -1:
                J = int(np.round(np.log2(self.n0 * self.dt / s0) / self.dj))
            else:
                J = self.J
            # The scales as of Mallat 1999
            sj = s0 * 2 ** (np.arange(0, J + 1) * self.dj)
            # Fourier equivalent frequencies
            freqs = 1 / (self.wavelet.flambda() * sj)
        else:
            # The wavelet scales using custom frequencies.
            sj = 1 / (self.wavelet.flambda() * self.freqs)
            freqs = self.freqs
            J = self.J
        return sj, freqs, J

    def _get_psi_ft_bar(self) -> torch.Tensor:
        x = torch.randn(1, self.n0)
        sj, *_ = self._get_params()

        x_ft = torch.fft.fft(pow2pad(x), dim=-1)
        N = x_ft.shape[-1]
        ftfreqs = 2 * np.pi * np.fft.fftfreq(N, self.dt)

        sj_col = sj[:, np.newaxis]
        return torch.from_numpy(
            (sj_col * ftfreqs[1] * N) ** 0.5
            * np.conjugate(self.wavelet.psi_ft(sj_col * ftfreqs))
        ).unsqueeze(0)

    @property
    def n_scales(self) -> int:
        return self._get_params()[-1] + 1

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.n_scales, self.n0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if next_pow2(x.shape[-1]) != self.n0:
            raise IndexError(f"Next power of two greater than n0 ({self.n0})")

        x_ft = torch.fft.fft(pow2pad(x), dim=-1).unsqueeze(1)
        return torch.fft.ifft(x_ft * self.psi_ft_bar.type_as(x), dim=-1)


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    PLOT: bool = False
    SR: int = 16_000

    y, sr = librosa.load(librosa.ex("trumpet"), sr=SR, duration=1)
    y = torch.from_numpy(y).unsqueeze(0)

    transform = Cwt(next_pow2(y.shape[-1]))
    X = transform(y)

    assert X.shape == (y.shape[0], *transform.output_shape)

    if PLOT:
        # Magnitude
        plt.imshow(X[0].abs().numpy(), aspect="auto", cmap="turbo")
        plt.show()

        # Phase
        plt.imshow(X[0].angle().numpy(), aspect="auto", cmap="turbo")
        plt.show()
