"""Neuron dictionary: match each neuron against candidate features.

For each neuron we measure
- how well its on/off state matches each boolean event (agreement), and
- how well its value h_t is predicted by each running value (affine R^2),
  plus a greedy multi-feature fit,
pooled over a set of steps, so a high score means the neuron plays the same
role at every one of those steps.
"""

from dataclasses import dataclass, field

import torch as th

from .features import events, running_values
from .hooked import RNNCache


def _pool(t, steps):
    """[batch, seq_len, ...] -> [batch * len(steps), ...]"""
    return t[:, steps].flatten(0, 1)


def event_agreement(cache: RNNCache, evs: dict, steps) -> dict[str, th.Tensor]:
    """Fraction of (input, step) pairs where each neuron's on/off equals the event: name -> [hidden]."""
    mask = _pool(cache.mask, steps).cpu()
    return {name: (mask == _pool(e, steps).cpu()[:, None]).float().mean(0) for name, e in evs.items()}


class ValueFitter:
    """Affine least-squares fits of neuron values on subsets of features, via a shared Gram matrix."""

    def __init__(self, cache: RNNCache, values: dict, steps):
        self.names = list(values)
        F = th.stack([_pool(v, steps).double().cpu() for v in values.values()], 1)
        Y = _pool(cache.h, steps).double().cpu()
        X = th.cat([F, th.ones_like(F[:, :1])], 1)  # last column is the intercept
        self.n = len(X)
        self.G = X.T @ X
        self.C = X.T @ Y  # [n_features + 1, hidden]
        self.y_mean = Y.mean(0)
        self.ss_tot = ((Y - self.y_mean) ** 2).sum(0)
        self.yy = (Y**2).sum(0)

    def fit(self, neuron, feature_idx):
        """Coefficients (features..., intercept) and R^2 for one neuron on a feature subset."""
        cols = list(feature_idx) + [len(self.names)]
        G = self.G[cols][:, cols]
        c = self.C[cols, neuron]
        coef = th.linalg.lstsq(G, c[:, None]).solution[:, 0]
        ss_res = self.yy[neuron] - 2 * coef @ c + coef @ G @ coef
        if self.ss_tot[neuron] <= 1e-12:
            return coef, float("nan")
        return coef, (1 - ss_res / self.ss_tot[neuron]).item()

    def single_r2(self):
        """R^2 of every single-feature fit: [n_features, hidden]."""
        hidden = self.C.shape[1]
        return th.tensor([[self.fit(n, [f])[1] for n in range(hidden)] for f in range(len(self.names))])

    def greedy(self, neuron, k):
        """Forward selection of up to k features; returns (names, coefficients, R^2)."""
        chosen, best_r2, best_coef = [], float("-inf"), None
        for _ in range(k):
            best = None
            for f in range(len(self.names)):
                if f in chosen:
                    continue
                coef, r2 = self.fit(neuron, chosen + [f])
                if r2 == r2 and (best is None or r2 > best[1]):
                    best = (f, r2, coef)
            if best is None or best[1] <= best_r2 + 1e-4:
                break
            chosen.append(best[0])
            best_r2, best_coef = best[1], best[2]
        return [self.names[f] for f in chosen], best_coef, best_r2


@dataclass
class NeuronEntry:
    neuron: int
    p_active: float
    best_event: str
    event_agreement: float
    best_value: str
    value_r2: float
    value_slope: float
    value_intercept: float
    multi_features: list[str]
    multi_r2: float
    label: str
    extra: dict = field(default_factory=dict)


def build_dictionary(cache: RNNCache, steps=None, *, multi_k=3, event_threshold=0.98,
                     value_threshold=0.95, always_on_threshold=0.95):
    """Dictionary entries for every neuron, and the full single-feature R^2 table."""
    x = cache.x.cpu()
    seq_len, hidden = cache.h.shape[1:]
    if steps is None:
        steps = list(range(1, seq_len))  # step 0 is trivially sign(W_ih) * sign(x_0)
    evs = events(x)
    vals = running_values(x)
    agree = event_agreement(cache, evs, steps)
    agree_mat = th.stack(list(agree.values()))  # [n_events, hidden]
    fitter = ValueFitter(cache, vals, steps)
    r2 = fitter.single_r2().nan_to_num(nan=0.0)  # [n_values, hidden]
    p_active = _pool(cache.mask, steps).float().mean(0).cpu()

    entries = []
    event_names = list(agree)
    for n in range(hidden):
        e = int(agree_mat[:, n].argmax())
        v = int(r2[:, n].argmax())
        coef, _ = fitter.fit(n, [v])
        multi, _, multi_r2 = fitter.greedy(n, multi_k)
        labels = []
        if agree_mat[e, n] >= event_threshold:
            labels.append(f"event {event_names[e]}")
        if r2[v, n] >= value_threshold:
            labels.append(f"value {fitter.names[v]}")
        if p_active[n] >= always_on_threshold:
            labels.append("always on")
        elif p_active[n] == 0:
            labels.append("dead")
        entries.append(NeuronEntry(
            neuron=n,
            p_active=p_active[n].item(),
            best_event=event_names[e],
            event_agreement=agree_mat[e, n].item(),
            best_value=fitter.names[v],
            value_r2=r2[v, n].item(),
            value_slope=coef[0].item(),
            value_intercept=coef[1].item(),
            multi_features=multi,
            multi_r2=multi_r2,
            label="; ".join(labels) or "unexplained",
        ))
    return entries, fitter.names, r2
