"""Scoring estimates of a model's error rate by mean squared error versus compute.

ARC's challenge (docs/PLAN.md, steps 9-13): a mechanistic estimate of a
model's accuracy should match random sampling in mean squared error for a
given compute, measured over noisy copies of the model. This module provides
the pieces:

- `noisy_family`: copies of a model with relative Gaussian noise on every
  weight, `w + sigma * |w| * eps`;
- `true_error`: a high-precision sampled error rate, the ground truth;
- `sampling_estimate`: the baseline estimator (E0);
- `score`: MSE of an estimator over a family, with the ground truth's own
  sampling variance subtracted.

Compute is counted in multiply-adds (MACs). Every estimator takes
(model, budget, seed) and returns (estimate, macs), where `budget` is the
estimator's own size knob.
"""

import copy
from dataclasses import dataclass

import torch as th
from alg_zoo.tasks import task_2nd_argmax

from .data import randn_inputs


def forward_macs(model):
    """Multiply-adds in one forward pass: input and recurrent weights per step, then the readout."""
    H, T = model.hidden_size, model.seq_len
    return T * (H + H * H) + H * model.linear.out_features


def perturb(model, sigma, seed):
    """A float64 copy of `model` with every weight w replaced by w + sigma * |w| * eps, eps ~ N(0, 1)."""
    out = copy.deepcopy(model).double().eval()
    gen = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in out.parameters():
            p.add_(sigma * p.abs() * th.randn(p.shape, generator=gen, dtype=p.dtype))
    return out


@dataclass
class Member:
    sigma: float
    index: int
    model: object

    @property
    def key(self):
        return f"sigma={self.sigma:g}/i={self.index}"


def noisy_family(model, sigmas, n_per, seed=0):
    """The unperturbed model (sigma 0) plus n_per noisy copies at each sigma."""
    members = [Member(0.0, 0, model.double().eval())]
    for s in sigmas:
        for i in range(n_per):
            members.append(Member(s, i, perturb(model, s, seed=seed + 1000 * i + round(s * 1e6))))
    return members


def errors_on(model, x, batch=2**18):
    """1.0 where the model's prediction is wrong. Batched: a forward pass keeps every
    hidden state, so 2^22 inputs at once would take over 10 GB for 32x10."""
    with th.no_grad():
        return th.cat([(model(xb).argmax(-1) != task_2nd_argmax(xb)).double() for xb in x.split(batch)])


def true_error(model, n, *, seed, batch=2**20):
    """Sampled error rate on n float64 randn inputs, and its variance p(1 - p) / n."""
    wrong = 0.0
    for b in range(0, n, batch):
        x = randn_inputs(min(batch, n - b), model.seq_len, seed=seed + b // batch, dtype=th.float64)
        wrong += errors_on(model, x).sum().item()
    p = wrong / n
    return p, p * (1 - p) / n


def sampling_estimate(model, budget, seed):
    """E0: the error rate on `budget` random inputs."""
    x = randn_inputs(budget, model.seq_len, seed=seed, dtype=th.float64)
    return errors_on(model, x).mean().item(), budget * forward_macs(model)


def score(estimates, truths, truth_vars):
    """MSE of estimates against the ground truth, minus the ground truth's own sampling variance.

    E[(est - truth_sampled)^2] = E[(est - truth)^2] + Var(truth_sampled) when the
    two are independent, so subtracting the known variance leaves the estimator's MSE.
    Returns (corrected MSE, the variance floor that was subtracted).
    """
    e, t, v = (th.tensor(a, dtype=th.float64) for a in (estimates, truths, truth_vars))
    raw = ((e - t) ** 2).mean().item()
    floor = v.mean().item()
    return raw - floor, floor
