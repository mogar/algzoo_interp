"""Candidate features of the input sequence, for labelling neurons.

Every feature is defined at each step t *after* reading x_t, so it lines up
with h[:, t] from run_with_cache. All functions take x of shape
[batch, seq_len] and return dicts of [batch, seq_len] tensors.

"max0" means a running max that also includes 0, i.e. max(0, x_0, ..., x_t).
ReLU units with no bias compare against 0 for free, so trained and
handcrafted models both tend to use this convention.
"""

import torch as th


def lagged(x, k):
    """x_{t-k} at each step t, zero before the sequence starts."""
    if k == 0:
        return x
    return th.cat([th.zeros_like(x[:, :k]), x[:, :-k]], dim=1)


def running_top2(x):
    """Largest and 2nd-largest of x_0..x_t at each t. The 2nd is -inf at t=0."""
    top1 = x.cummax(1).values
    top2 = th.full_like(x, float("-inf"))
    for t in range(1, x.shape[1]):
        top2[:, t] = x[:, : t + 1].topk(2, dim=1).values[:, 1]
    return top1, top2


def max0_excluding_lag(x, k):
    """max(0, {x_j : j <= t, j != t-k}) at each step t."""
    out = th.zeros_like(x)
    for t in range(x.shape[1]):
        vals = x[:, : t + 1].clone()
        if t - k >= 0:
            vals[:, t - k] = float("-inf")
        out[:, t] = vals.amax(1).clamp(min=0)
    return out


def steps_since(event):
    """Steps since the event last fired, counting the current step as 0.

    Before the event has ever fired, counts from the start as if it fired at t=-1.
    """
    out = th.zeros_like(event, dtype=th.float32)
    last = th.full_like(event[:, 0], -1, dtype=th.float32)
    for t in range(event.shape[1]):
        last = th.where(event[:, t], th.full_like(last, t), last)
        out[:, t] = t - last
    return out


def events(x):
    """Boolean events a 2nd argmax algorithm might detect.

    "_pos" variants also require x_t > 0 (compare against max0).
    """
    top1, top2 = running_top2(x)
    out = {
        "new_max": x >= top1,
        "new_max_pos": (x >= top1) & (x > 0),
        "new_top2": x >= top2,
        "new_top2_pos": (x >= top2) & (x > 0),
    }
    for k in range(x.shape[1]):
        # Zero padding makes this False before the sequence starts
        out[f"x_{{t-{k}}}>0"] = lagged(x, k) > 0
    return out


def running_values(x):
    """Continuous running quantities a neuron might store."""
    seq_len = x.shape[1]
    top1, top2 = running_top2(x)
    max0 = top1.clamp(min=0)
    max0_prev = lagged(max0, 1)
    second = th.where(th.isinf(top2), th.zeros_like(top2), top2)
    ev = events(x)
    out = {}
    for k in range(seq_len):
        out[f"x_{{t-{k}}}"] = lagged(x, k)
        out[f"relu(x_{{t-{k}}})"] = lagged(x, k).clamp(min=0)
    out.update(
        {
            "max_so_far": top1,
            "2nd_max_so_far": second,
            "max-2nd": top1 - second,
            "max-x_t": top1 - x,
            "max0": max0,
            "max0_prev": max0_prev,
            "max0-max0_prev": max0 - max0_prev,
            "steps_since_new_max_pos": steps_since(ev["new_max_pos"]),
            "steps_since_new_top2_pos": steps_since(ev["new_top2_pos"]),
        }
    )
    # max0 over everything so far except one earlier input, minus the current
    # input; k=0 is plain max0 - x_t (handcrafted neurons 10-18 and 21)
    out["max0-x_t"] = max0 - x
    for k in range(1, seq_len):
        out[f"max0_excl(t-{k})-x_t"] = max0_excluding_lag(x, k) - x
    return out
