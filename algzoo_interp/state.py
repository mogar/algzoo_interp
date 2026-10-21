"""Compare hidden states with the state of an exact online 2nd argmax algorithm.

An exact online algorithm only needs to remember, after reading x_t:
the largest and 2nd-largest values so far, and how many steps ago each
arrived (their lags). This module computes that "ideal state" and measures
how much of each neuron it explains, using a flexible fit: one group per
(step, lag of max, lag of 2nd max) cell, and within each cell a linear fit on
continuous features (the values, the current input, ...).
"""

import torch as th

from .features import lagged, running_top2


def ideal_state(x):
    """Ideal online state after each step: dict of [batch, seq_len] tensors.

    lag_max / lag_2nd: steps since the max / 2nd max so far arrived (lag_2nd is
    0 at t=0, where there is no 2nd value).
    """
    batch, seq_len = x.shape
    top1, top2 = running_top2(x)
    top2 = th.where(th.isinf(top2), th.zeros_like(top2), top2)
    lag_max = th.zeros(batch, seq_len, dtype=th.long, device=x.device)
    lag_2nd = th.zeros_like(lag_max)
    for t in range(seq_len):
        order = x[:, : t + 1].argsort(-1, descending=True)
        lag_max[:, t] = t - order[:, 0]
        if t > 0:
            lag_2nd[:, t] = t - order[:, 1]
    return {"max": top1, "2nd": top2, "lag_max": lag_max, "lag_2nd": lag_2nd}


def continuous_features(x, state, level):
    """Continuous features for the within-cell fits, by level.

    "none": intercept only (positions alone).
    "values": + the max and 2nd max so far (raw and clipped at 0).
    "values+input": + the current input and how far it falls below the
        previous max / 2nd max (clipped at 0).
    """
    feats = []
    if level in ("values", "values+input"):
        feats += [state["max"], state["2nd"], state["max"].clamp(min=0), state["2nd"].clamp(min=0)]
    if level == "values+input":
        prev_max0 = lagged(state["max"], 1).clamp(min=0)
        prev_2nd0 = lagged(state["2nd"], 1).clamp(min=0)
        feats += [x, x.clamp(min=0), (prev_max0 - x).clamp(min=0), (prev_2nd0 - x).clamp(min=0)]
    return feats


def cellwise_r2(values, cells, feats):
    """R^2 of per-cell affine fits.

    values: [n, k] targets (e.g. neurons); cells: [n] integer cell ids;
    feats: list of [n] continuous features (may be empty).
    Returns [k].
    """
    _, cells = th.unique(cells, return_inverse=True)
    n_cells = int(cells.max()) + 1
    X = th.stack([th.ones_like(values[:, 0])] + list(feats), -1).double()
    Y = values.double()
    k = X.shape[1]
    XtX = th.zeros(n_cells, k, k, dtype=th.double).index_add_(0, cells, X[:, :, None] * X[:, None, :])
    XtY = th.zeros(n_cells, k, Y.shape[1], dtype=th.double).index_add_(0, cells, X[:, :, None] * Y[:, None, :])
    # Tiny ridge keeps cells with few inputs or constant features solvable
    coef = th.linalg.lstsq(XtX + 1e-9 * th.eye(k, dtype=th.double), XtY).solution
    pred = (X[:, :, None] * coef[cells]).sum(1)
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def state_r2(cache, steps, level):
    """How much of each neuron's value the ideal state explains: [hidden]."""
    x = cache.x.cpu()
    state = ideal_state(x)
    seq_len = x.shape[1]
    t = th.arange(seq_len).expand_as(state["lag_max"])
    cells = (t * seq_len + state["lag_max"]) * seq_len + state["lag_2nd"]
    pick = lambda a: a[:, steps].reshape(-1)  # noqa: E731
    feats = [pick(f) for f in continuous_features(x, state, level)]
    h = cache.h.cpu()[:, steps].reshape(-1, cache.h.shape[-1])
    return cellwise_r2(h, pick(cells), feats)
