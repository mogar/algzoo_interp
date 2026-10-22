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
  sampling variance subtracted;
- `Stratified`: E1, exact cell probabilities times the error sampled within
  each cell (algzoo_interp.order_stats);
- `CellLines`: E2, the same cells, with the error in each cell integrated
  exactly along a tie line where that is cheaper (algzoo_interp.lines).

Compute is counted in multiply-adds (MACs). Every estimator takes
(model, budget, seed) and returns (estimate, macs), where `budget` is the
estimator's own size knob.
"""

import copy
from dataclasses import dataclass

import torch as th
from alg_zoo.tasks import task_2nd_argmax

from .data import randn_inputs
from .lines import cell_interval, line_error_mass, tie_lines
from .order_stats import sample_cell


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


class Stratified:
    """E1: Σ_c P(c) × (error rate on n_c inputs sampled within cell c).

    P(c) is exact (algzoo_interp.order_stats) and only the error within each cell
    is sampled. The budget is split across cells in proportion to `weights`;
    every cell gets at least one input, so the estimate is unbiased for any
    weights. Neyman allocation, weights ∝ P(c) sqrt(p_c (1 - p_c)), minimises
    the variance.
    """

    def __init__(self, cells, probs, weights, n):
        self.cells, self.probs, self.n = cells, probs, n
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def allocation(self, budget):
        return [max(1, int(budget * w)) for w in self.weights]

    def __call__(self, model, budget, seed):
        gen = th.Generator().manual_seed(seed)
        est, used = 0.0, 0
        for c, P, m in zip(self.cells, self.probs, self.allocation(budget)):
            est += P * errors_on(model, sample_cell(c, self.n, m, gen)).mean().item()
            used += m
        return est, used * forward_macs(model)


def neyman_weights(model, cells, probs, n, per_cell, seed=0):
    """Pilot error rate in each cell (add-one smoothed) and the Neyman weights from it."""
    gen = th.Generator().manual_seed(seed)
    rates, weights = [], []
    for c, P in zip(cells, probs):
        k = errors_on(model, sample_cell(c, n, per_cell, gen)).sum().item()
        p = (k + 1) / (per_cell + 2)
        rates.append(k / per_cell)
        weights.append(P * (p * (1 - p)) ** 0.5)
    return rates, weights



# The true answer along a tie line, for s > 0 and s < 0 (x_j > x_k at s > 0):
# for ranks (1, 2) the 2nd largest is the larger of the pair, for (0, 1) the smaller.
def _truth(ranks, j, k):
    return (j, k) if ranks == (1, 2) else (k, j)


def cell_values(model, x, cell, method):
    """Per-input values whose mean, over x sampled in `cell`, is the cell's error rate.

    method "sample": the 0/1 error at x (cost 1 forward pass each).
    method (1, 2) or (0, 1): the exact error rate along the tie line through x
    between the inputs at those ranks, over the part of the line inside the
    cell, |s| in [t_lo, t_hi] (cost 2 forward passes per walk step). Given the
    line's y, x's own s is N(0, 1) restricted to that part, so this is unbiased.
    Returns (values, cost in forward passes).
    """
    if method == "sample":
        return errors_on(model, x), float(len(x))
    y, u, s, w, mid, others, j, k = tie_lines(x, method)
    lo, hi = cell_interval(cell, method, mid, others, w)
    pos, neg = _truth(method, j, k)
    m1, p1 = line_error_mass(model, y, u, lo, hi, pos, neg)
    m2, p2 = line_error_mass(model, y, u, -hi, -lo, pos, neg)
    denom = 2 * (th.special.ndtr(-lo) - th.special.ndtr(-hi))
    return (m1 + m2) / denom, 2.0 * (p1 + p2).sum().item()


class CellLines:
    """E2: E1's cells, with each cell's error measured by its cheapest method.

    Each cell uses 0/1 samples or exact integration along one of the tie lines
    (`cell_values`), whichever has the smaller σ √cost in the pilot. Inputs are
    allocated by Neyman allocation with costs, n_c ∝ P(c) σ_c / √cost_c, which
    minimises the variance for a given total compute. Every cell gets at least
    one input, so the estimate is unbiased for any design.
    """

    def __init__(self, cells, probs, methods, alloc, n):
        self.cells, self.probs, self.methods, self.n = cells, probs, methods, n
        total = sum(alloc)
        self.alloc = [a / total for a in alloc]

    def __call__(self, model, budget, seed):
        """`budget` is in forward passes; the result's compute is in multiply-adds."""
        gen = th.Generator().manual_seed(seed)
        est, cost = 0.0, 0.0
        for c, P, meth, a in zip(self.cells, self.probs, self.methods, self.alloc):
            m = max(1, int(budget * a))
            v, k = cell_values(model, sample_cell(c, self.n, m, gen), c, meth)
            est += P * v.mean().item()
            cost += k
        return est, cost * forward_macs(model)


def design_cell_lines(model, cells, probs, n, per_cell, seed=0, margin=0.8):
    """Pilot every method in every cell and pick each cell's cheapest.

    A line method replaces 0/1 samples only if its σ √cost is below `margin`
    times theirs. Line values are heavy-tailed: a few lines are wrong along
    their whole segment, and a small pilot can miss them entirely and
    underestimate σ.

    Returns (methods, allocation weights, rows describing each cell, and the
    predicted variance of E2 per unit compute relative to plain sampling).
    """
    gen = th.Generator().manual_seed(seed)
    methods, alloc, rows = [], [], []
    for c, P in zip(cells, probs):
        x = sample_cell(c, n, per_cell, gen)
        options = {}
        for meth in ["sample"] + ([(1, 2), (0, 1)] if c.kind == "pos" else []):
            v, k = cell_values(model, x, c, meth)
            if meth == "sample":
                p = (v.sum().item() + 1) / (per_cell + 2)
                sd = (p * (1 - p)) ** 0.5
            else:
                sd = max(v.std().item(), 1e-9)
            options[meth] = (sd, k / per_cell, v.mean().item())
        work = {mth: sd * cost**0.5 for mth, (sd, cost, _) in options.items()}
        best = min(work, key=work.get)
        if best != "sample" and work[best] >= margin * work["sample"]:
            best = "sample"
        sd, cost, mean = options[best]
        methods.append(best)
        alloc.append(P * sd / cost**0.5)
        rows.append({"cell": c.name, "prob": P, "method": str(best), "pilot_error": mean, "sd": sd, "cost": cost,
                     "options": {str(k): v for k, v in options.items()}})
    total = sum(r["prob"] * r["pilot_error"] for r in rows)
    work = sum(r["prob"] * r["sd"] * r["cost"] ** 0.5 for r in rows) ** 2  # variance × compute
    return methods, alloc, rows, total * (1 - total) / work
