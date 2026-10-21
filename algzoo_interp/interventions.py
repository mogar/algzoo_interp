"""Ablation and activation patching on hidden states h_t.

Interventions replace h_t after the ReLU, so their effects propagate through
all later steps. neurons and steps select what is replaced; None means all.
"""

import torch as th

from .hooked import RNNCache, run_with_cache


def _as_list(idx, size):
    if idx is None:
        return list(range(size))
    if isinstance(idx, int):
        return [idx]
    return list(idx)


def replace_h(model, x, values, neurons=None, steps=None) -> RNNCache:
    """Run on x with h[:, t, neurons] replaced by values[..., t, neurons] for t in steps.

    values has shape [seq_len, hidden] (same for every input) or
    [batch, seq_len, hidden] (per input).
    """
    hidden_size = model.hidden_size
    neurons = _as_list(neurons, hidden_size)
    steps = set(_as_list(steps, x.shape[1]))
    values = values.to(device=model.device, dtype=model.dtype)

    def hook(t, h):
        if t not in steps:
            return h
        h = h.clone()
        h[:, neurons] = values[..., t, :][..., neurons]
        return h

    return run_with_cache(model, x, hooks={"h": hook})


@th.no_grad()
def ablate(model, x, neurons=None, steps=None, mode="zero", ref_x=None, seed=0):
    """Ablate neurons at the given steps.

    mode:
    - "zero": set to 0
    - "mean": set to the mean of h_t over ref_x (default x), per step
    - "resample": set to h_t from a different input in ref_x (default x,
      randomly permuted), per input
    """
    seq_len, hidden_size = x.shape[1], model.hidden_size
    if mode == "zero":
        values = th.zeros(seq_len, hidden_size)
    elif mode == "mean":
        values = run_with_cache(model, x if ref_x is None else ref_x).h.mean(0)
    elif mode == "resample":
        if ref_x is None:
            gen = th.Generator().manual_seed(seed)
            ref_x = x[th.randperm(len(x), generator=gen).to(x.device)]
        assert ref_x.shape == x.shape, "resample ablation needs ref_x shaped like x"
        values = run_with_cache(model, ref_x).h
    else:
        raise ValueError(f"Unknown ablation mode {mode!r}")
    return replace_h(model, x, values, neurons=neurons, steps=steps)


@th.no_grad()
def patch(model, x, x_src, neurons=None, steps=None) -> RNNCache:
    """Run on x, patching in h values from the run on x_src (same shape as x)."""
    return replace_h(model, x, run_with_cache(model, x_src).h, neurons=neurons, steps=steps)
