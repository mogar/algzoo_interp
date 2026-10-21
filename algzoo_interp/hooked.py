"""Hand-unrolled RNN that exposes every intermediate value.

alg_zoo's RNNs compute, with no biases,

    h_0 = 0
    pre_t = W_ih * x_t + W_hh @ h_{t-1}
    h_t = ReLU(pre_t)
    logits = W_out @ h_T

Time steps are 0-indexed throughout: x[:, t] produces pre[:, t] and h[:, t].
"""

from dataclasses import dataclass
from typing import Callable

import torch as th

# A hook receives (t, value) with value of shape [batch, hidden] and returns
# the replacement value
Hook = Callable[[int, th.Tensor], th.Tensor]


@dataclass
class Weights:
    W_ih: th.Tensor  # [hidden]
    W_hh: th.Tensor  # [hidden, hidden]
    W_out: th.Tensor  # [n_out, hidden]


def get_weights(model) -> Weights:
    return Weights(
        W_ih=model.rnn.weight_ih_l0[:, 0],
        W_hh=model.rnn.weight_hh_l0,
        W_out=model.linear.weight,
    )


@dataclass
class RNNCache:
    x: th.Tensor  # [batch, seq_len]
    pre: th.Tensor  # [batch, seq_len, hidden]
    h: th.Tensor  # [batch, seq_len, hidden]
    logits: th.Tensor  # [batch, n_out]

    @property
    def mask(self) -> th.Tensor:
        """Which neurons are active (pre > 0) at each step: [batch, seq_len, hidden]."""
        return self.pre > 0


def run_with_cache(model, x, hooks: dict[str, Hook] | None = None) -> RNNCache:
    """Run the model step by step, caching pre-activations and hidden states.

    hooks may contain "pre" (applied to pre_t before the ReLU) and/or "h"
    (applied to h_t after the ReLU). Cached values are the hooked ones.
    """
    hooks = hooks or {}
    w = get_weights(model)
    x = x.to(device=w.W_ih.device, dtype=w.W_ih.dtype)
    h = x.new_zeros(x.shape[0], w.W_ih.shape[0])
    pres, hs = [], []
    for t in range(x.shape[1]):
        pre = x[:, t, None] * w.W_ih + h @ w.W_hh.T
        if "pre" in hooks:
            pre = hooks["pre"](t, pre)
        h = th.relu(pre)
        if "h" in hooks:
            h = hooks["h"](t, h)
        pres.append(pre)
        hs.append(h)
    return RNNCache(
        x=x, pre=th.stack(pres, dim=1), h=th.stack(hs, dim=1), logits=h @ w.W_out.T
    )


def logit_contributions(model, cache: RNNCache) -> th.Tensor:
    """Each neuron's direct contribution to each final logit: [batch, n_out, hidden].

    Sums over the last axis to cache.logits.
    """
    W_out = get_weights(model).W_out
    return cache.h[:, -1, None, :] * W_out
