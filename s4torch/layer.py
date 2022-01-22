"""

    S4 Layer

"""
from typing import Callable, Union

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


def _make_omega_l(l_max: int) -> torch.Tensor:
    return torch.arange(l_max).type(torch.complex128).mul(2j * np.pi / l_max).exp()


def _make_hippo(N: int) -> np.ndarray:
    def idx2value(n: int, k: int) -> Union[int, float]:
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

    lambda_, V = np.linalg.eig(S)
    return lambda_, p, q, V


def _make_s4_buffers(n: int) -> list[torch.Tensor]:
    lambda_, p, q, V = _make_nplr_hippo(n)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    return [torch.from_numpy(i) for i in (p, q, lambda_)]


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
    lambda_: torch.Tensor,
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

        k00 = _cauchy_dot(a0 * b0, g=g, lambd=lambda_)
        k01 = _cauchy_dot(a0 * b1, g=g, lambd=lambda_)
        k10 = _cauchy_dot(a1 * b0, g=g, lambd=lambda_)
        k11 = _cauchy_dot(a1 * b1, g=g, lambd=lambda_)
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    l_max = u.shape[1]
    ud = rfft(F.pad(u, pad=(0, 0, 0, l_max, 0, 0)), dim=1)
    Kd = rfft(F.pad(K, pad=(0, l_max)), dim=-1)
    return irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1)


class S4Layer(nn.Module):
    p: torch.Tensor
    q: torch.Tensor
    lambda_: torch.Tensor
    omega_l: torch.Tensor
    ifft_order: torch.Tensor

    def __init__(
        self,
        d_model: int,
        n: int,
        l_max: int,
        train_p: bool = False,
        train_q: bool = False,
        train_lambda: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.train_p = train_p
        self.train_q = train_q
        self.train_lambda = train_lambda

        p, q, lambda_ = _make_s4_buffers(n)
        self._register_tensor("p", tensor=p, trainable=train_p)
        self._register_tensor("q", tensor=q, trainable=train_q)
        self._register_tensor("lambda_", tensor=lambda_, trainable=train_lambda)
        self._register_tensor(
            "omega_l",
            tensor=_make_omega_l(self.l_max),
            trainable=False,
        )

        self._register_tensor(
            "ifft_order",
            tensor=torch.as_tensor(
                [i if i == 0 else self.l_max - i for i in range(self.l_max)],
                dtype=torch.long,
            ),
            trainable=False,
        )

        self.B = nn.Parameter(init.xavier_normal_(torch.empty(n, d_model)).T)
        self.Ct = nn.Parameter(init.xavier_normal_(torch.empty(d_model, n)))
        self.D = nn.Parameter(torch.ones(d_model))[None, None, ...]
        self.log_step = nn.Parameter(_log_step_initializer(torch.rand(d_model)))

    def _register_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        trainable: bool,
    ) -> None:
        if trainable:
            self.register_parameter(name, param=tensor)
        else:
            self.register_buffer(name, tensor=tensor)

    def _conv_from_gen(
        self,
        k_gen: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        at_roots = k_gen(self.omega_l)
        out = ifft(at_roots, n=self.l_max, dim=-1)
        return torch.stack([i[self.ifft_order] for i in out]).real.float()

    @property
    def K(self) -> torch.Tensor:  # noqa
        k_gen = _k_gen_dplr(
            lambda_=self.lambda_,
            p=self.p,
            q=self.q,
            B=self.B,
            Ct=self.Ct,
            step=self.log_step.exp(),
        )
        return self._conv_from_gen(k_gen).unsqueeze(0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _non_circular_convolution(u, K=self.K) + (self.D * u)


if __name__ == "__main__":
    N = 32
    d_model = 128
    l_max = 784

    u = torch.randn((1, l_max, d_model))

    s4_layer = S4Layer(d_model, n=N, l_max=l_max)
    assert s4_layer(u).shape == u.shape
