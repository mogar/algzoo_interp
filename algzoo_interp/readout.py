"""Readout analyses shared by scripts/readout_analysis.py and scripts/readout_generality.py.

See docs/HYPOTHESES.md §G for what these measure.
"""

import torch as th


def additive_codes(h, p1, p2, T):
    """Weighted least-squares fit of per-cell mean states: mu(i, j) ≈ a(i) + b(j)."""
    cell = p1 * T + p2
    cnt = th.bincount(cell, minlength=T * T).double()
    mu = th.zeros(T * T, h.shape[1], dtype=th.float64).index_add_(0, cell, h) / cnt.clamp(min=1)[:, None]
    I, J = th.arange(T * T) // T, th.arange(T * T) % T
    cells = th.where(I != J)[0]
    X = th.zeros(len(cells), 2 * T, dtype=th.float64)
    X[th.arange(len(cells)), I[cells]] = 1
    X[th.arange(len(cells)), T + J[cells]] = 1
    w = cnt[cells].sqrt()[:, None]
    coef = th.linalg.lstsq(X * w, mu[cells] * w).solution
    fit = X @ coef
    wmean = (cnt[cells][:, None] * mu[cells]).sum(0) / cnt[cells].sum()
    r2 = 1 - (cnt[cells][:, None] * (mu[cells] - fit) ** 2).sum() / (cnt[cells][:, None] * (mu[cells] - wmean) ** 2).sum()
    return coef[:T], coef[T:], r2.item()


def greedy(h, h_src, W, target, ok, start, steps):
    """Greedily add neurons to the patched set, maximising how often the prediction moves to `target`."""
    chosen, curve = list(start), []
    for _ in range(steps):
        best = None
        for n in range(h.shape[1]):
            if n in chosen:
                continue
            hp = h.clone()
            hp[:, chosen + [n]] = h_src[:, chosen + [n]]
            v = (((hp @ W.T).argmax(-1) == target)[ok]).double().mean().item()
            if best is None or v > best[0]:
                best = (v, n)
        chosen.append(best[1])
        curve.append({"add": best[1], "moved": best[0]})
    return curve


def membership_flags(cache):
    """Best single-neuron on/off flag, per lag k, for "the input k steps ago is in the running top two".

    Pooled over steps. Returns {k: {neuron, polarity, p_on_if_in_top2, p_on_if_not, accuracy, majority_baseline}}.
    """
    x = cache.x.cpu()
    T, hidden = x.shape[1], cache.h.shape[-1]
    flags = {}
    mask = cache.mask.double()
    for k in range(1, T):
        on_in, n_in = th.zeros(hidden, dtype=th.float64), 0
        on_out, n_out = th.zeros(hidden, dtype=th.float64), 0
        for t in range(k, T):
            o = x[:, : t + 1].argsort(-1, descending=True)
            inn = (o[:, 0] == t - k) | (o[:, 1] == t - k)
            on_in += mask[inn, t].sum(0)
            n_in += int(inn.sum())
            on_out += mask[~inn, t].sum(0)
            n_out += int((~inn).sum())
        p_in, p_out = on_in / n_in, on_out / max(n_out, 1)
        frac = n_in / (n_in + n_out)
        acc_on = frac * p_in + (1 - frac) * (1 - p_out)
        acc = th.maximum(acc_on, 1 - acc_on)
        best = int(acc.argmax())
        flags[k] = {"neuron": best, "polarity": "on-when-in" if acc_on[best] >= 0.5 else "off-when-in",
                    "p_on_if_in_top2": p_in[best].item(), "p_on_if_not": p_out[best].item(),
                    "accuracy": acc[best].item(), "majority_baseline": max(frac, 1 - frac)}
    return flags
