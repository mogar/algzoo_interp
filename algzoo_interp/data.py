"""Input distributions: the training distribution plus controlled probes.

All functions return float tensors of shape [n, seq_len].
"""

import torch as th


def randn_inputs(n, seq_len, *, seed=0, device="cpu", dtype=th.float32):
    """Standard normal inputs, matching the training distribution."""
    gen = th.Generator(device=device).manual_seed(seed)
    return th.randn(n, seq_len, generator=gen, device=device, dtype=dtype)


def with_top2_positions(n, seq_len, argmax_pos, second_pos, *, seed=0, **kwargs):
    """randn inputs conditioned on the positions of the largest and 2nd-largest values.

    Samples iid normals, then moves the largest to argmax_pos and the 2nd largest
    to second_pos, keeping the rest in their original order. Since the remaining
    values are exchangeable, this is exactly the conditional distribution.
    """
    assert argmax_pos != second_pos
    x = randn_inputs(n, seq_len, seed=seed, **kwargs)
    order = x.argsort(-1, descending=True)
    top, second = order[:, :1], order[:, 1:2]
    rest_mask = th.ones_like(x, dtype=th.bool)
    rest_mask.scatter_(1, order[:, :2], False)
    rest = x[rest_mask].reshape(n, seq_len - 2)
    out = th.empty_like(x)
    other_pos = [p for p in range(seq_len) if p not in (argmax_pos, second_pos)]
    out[:, other_pos] = rest
    out[:, argmax_pos] = x.gather(1, top)[:, 0]
    out[:, second_pos] = x.gather(1, second)[:, 0]
    return out


def with_top2_gap(n, seq_len, gap, *, seed=0, **kwargs):
    """randn inputs with the largest value moved to exactly (2nd largest + gap).

    Small gaps give near-ties between the top two.
    """
    x = randn_inputs(n, seq_len, seed=seed, **kwargs)
    order = x.argsort(-1, descending=True)
    second_val = x.gather(1, order[:, 1:2])
    return x.scatter(1, order[:, :1], second_val + gap)


def sorted_inputs(n, seq_len, *, descending=False, seed=0, **kwargs):
    """randn inputs sorted ascending (2nd argmax at seq_len - 2) or descending (at 1)."""
    x = randn_inputs(n, seq_len, seed=seed, **kwargs)
    return x.sort(-1, descending=descending).values


def vary_position(base, pos, values):
    """Copies of one base sequence [seq_len] with position pos set to each of values.

    Returns [len(values), seq_len], for sweeping a single input.
    """
    values = th.as_tensor(values, dtype=base.dtype, device=base.device)
    x = base.expand(len(values), -1).clone()
    x[:, pos] = values
    return x
