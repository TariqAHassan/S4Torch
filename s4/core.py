"""

    S4 PyTorch

"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.fft import ifft, irfft, rfft
from torch.nn import functional as F
from torch.nn import init


def _log_step_initializer(
    tensor: torch.Tensor,  # values should be from U(0, 1)
    dt_min: float = 0.001,
    dt_max: float = 0.1,
) -> torch.Tensor:
    scale = np.log(dt_max) - np.log(dt_min)
    return tensor * scale + np.log(dt_min)


def _make_hippo(N: int) -> np.ndarray:
    def idx2value(n: int, k: int) -> int | float:
        if n > k:
            return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    hippo = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hippo[i, j] = idx2value(i + 1, j + 1)
    return hippo


def _make_nplr_hippo(N: int) -> tuple[np.ndarray, ...]:
    nhippo = -1 * _make_hippo(N)

    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]

    lmbda, V = np.linalg.eig(S)
    return lmbda, p, q, V


def _make_s4_buffers(n: int) -> list[torch.Tensor]:
    lmbda, p, q, V = _make_nplr_hippo(n)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    return [torch.from_numpy(i) for i in (p, q, lmbda)]


def _cauchy_dot(
    v: torch.Tensor,
    g: torch.Tensor,
    lambd: torch.Tensor,
) -> torch.Tensor:
    if v.ndim == 1:
        v = v.unsqueeze(0).unsqueeze(0)
    elif v.ndim == 2:
        v = v.unsqueeze(1)
    elif v.ndim != 3:
        raise IndexError(f"Expected `v` to be 1D, 2D or 3D, got {v.ndim}D")
    denom = torch.stack([i - lambd[None, ...] for i in g[..., :, None]])
    return (v / denom).sum(dim=-1)


def _k_gen_dplr(
    Lambda: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    B: torch.Tensor,
    Ct: torch.Tensor,
    step: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a0, a1 = Ct.conj(), q.conj()
    b0, b1 = B, p

    def gen(omega: torch.Tensor) -> torch.Tensor:
        g = torch.outer(2.0 / step, (1.0 - omega) / (1.0 + omega))
        c = 2.0 / (1.0 + omega)

        k00 = _cauchy_dot(a0 * b0, g=g, lambd=Lambda)
        k01 = _cauchy_dot(a0 * b1, g=g, lambd=Lambda)
        k10 = _cauchy_dot(a1 * b0, g=g, lambd=Lambda)
        k11 = _cauchy_dot(a1 * b1, g=g, lambd=Lambda)
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


def _conv_from_gen(
    omega_l: torch.Tensor,
    k_gen: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    L = omega_l.shape[-1]
    at_roots = k_gen(omega_l)
    out = ifft(at_roots, n=L, dim=-1)
    order = torch.as_tensor(
        [i if i == 0 else L - i for i in range(L)],
        dtype=torch.long,
    )
    return torch.stack([i[order] for i in out]).real


def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    l_max = u.shape[1]
    assert K.shape[-1] == l_max

    ud = rfft(F.pad(u, pad=(0, 0, 0, l_max, 0, 0)), dim=1)
    Kd = rfft(F.pad(K, pad=(0, l_max)), dim=-1)
    return irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1)


class S4Layer(nn.Module):
    p: torch.Tensor
    q: torch.Tensor
    lmbda: torch.Tensor
    omega_l: torch.Tensor

    def __init__(self, n: int, d_model: int, l_max: int) -> None:
        super().__init__()
        self.n = n
        self.d_model = d_model
        self.l_max = l_max

        p, q, lmbda = _make_s4_buffers(n)
        self.register_buffer("p", p)
        self.register_buffer("q", q)
        self.register_buffer("lmbda", lmbda)
        self.register_buffer(
            "omega_l",
            torch.from_numpy(np.exp((2j * np.pi / self.l_max) * np.arange(self.l_max))),
        )

        self.B = nn.Parameter(init.xavier_normal_(torch.empty(n, d_model)).T)
        self.Ct = nn.Parameter(init.xavier_normal_(torch.empty(d_model, n)))
        self.D = nn.Parameter(torch.ones(d_model))[None, None, ...]
        self.log_step = nn.Parameter(_log_step_initializer(torch.rand(d_model)))

    @property
    def K(self) -> torch.Tensor:  # noqa
        k_gen = _k_gen_dplr(
            Lambda=self.lmbda,
            p=self.p,
            q=self.q,
            B=self.B,
            Ct=self.Ct,
            step=self.log_step.exp(),
        )
        return _conv_from_gen(self.omega_l, k_gen=k_gen).unsqueeze(0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _non_circular_convolution(u, K=self.K) + (self.D * u)


if __name__ == "__main__":
    N = 32
    d_model = 128
    l_max = 784

    u = torch.ones((1, l_max, d_model))

    self = S4Layer(
        n=N,
        d_model=d_model,
        l_max=l_max,
    )
    out = self(u)
    assert out.shape == u.shape
