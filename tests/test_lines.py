import pytest
import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import load_2nd_argmax, randn_inputs
from algzoo_interp.estimation import cell_values
from algzoo_interp.lines import cell_interval, line_error_mass, tie_lines
from algzoo_interp.order_stats import Cell, cell_index, gap_cells, sample_cell


@pytest.fixture(scope="module")
def model():
    return load_2nd_argmax(16, 10).double().eval()


def grid_mass(model, y, u, a, b, G=200_001):
    s = th.linspace(float(a), float(b), G, dtype=th.float64)
    x = y + s[:, None] * u
    with th.no_grad():
        wrong = (model(x).argmax(-1) != task_2nd_argmax(x)).double()
    phi = th.exp(-(s**2) / 2) / (2 * th.pi) ** 0.5
    return ((wrong * phi)[:-1] * (s[1] - s[0])).sum().item()


@pytest.mark.parametrize("ranks", [(1, 2), (0, 1)])
def test_tie_lines(ranks):
    x = randn_inputs(500, 10, seed=1, dtype=th.float64)
    y, u, s, w, mid, others, j, k = tie_lines(x, ranks)
    assert th.allclose(y + s[:, None] * u, x)
    assert ((y * u).sum(1).abs() < 1e-12).all()
    assert (s > 0).all() and (s < w).all()
    order = x.argsort(-1, descending=True)
    assert (order[:, ranks[0]] == j).all() and (order[:, ranks[1]] == k).all()


@pytest.mark.parametrize("ranks", [(1, 2), (0, 1)])
def test_line_error_mass_matches_grid(model, ranks):
    x = randn_inputs(6, 10, seed=2, dtype=th.float64)
    y, u, s, w, mid, others, j, k = tie_lines(x, ranks)
    pos, neg = (j, k) if ranks == (1, 2) else (k, j)
    mass, passes = line_error_mass(model, y, u, -w, w, pos, neg)
    assert (passes >= 1).all()
    for i in range(len(x)):
        # the grid's error is about (number of error boundaries) × step × φ
        step = 2 * w[i].item() / 200_000
        assert abs(mass[i].item() - grid_mass(model, y[i], u[i], -w[i], w[i])) < 20 * step


@pytest.mark.parametrize("ranks", [(1, 2), (0, 1)])
def test_cell_interval_stays_in_cell(ranks):
    cells = gap_cells([0, 1e-3, 0.03, 0.3, 1.0], [0, 1e-3, 0.03, 0.3, 1.0, float("inf")])
    gen = th.Generator().manual_seed(3)
    for idx, c in enumerate(cells):
        if c.kind != "pos":
            continue
        x = sample_cell(c, 10, 200, gen)
        y, u, s, w, mid, others, j, k = tie_lines(x, ranks)
        lo, hi = cell_interval(c, ranks, mid, others, w)
        assert ((s >= lo - 1e-12) & (s <= hi + 1e-12)).all()
        t = lo + th.rand(len(x), generator=gen, dtype=th.float64) * (hi - lo)
        for sign in (1, -1):
            assert (cell_index(y + (sign * t)[:, None] * u, cells) == idx).all(), c.name


def test_cell_values_unbiased(model):
    # a near-tie cell with a high error rate: line integrals and 0/1 samples agree on average
    c = Cell("pos", (0.3, 1.0), (1e-3, 3e-3))
    gen = th.Generator().manual_seed(4)
    x = sample_cell(c, 10, 4000, gen)
    bern, _ = cell_values(model, x, c, "sample")
    for ranks in [(1, 2), (0, 1)]:
        v, cost = cell_values(model, x, c, ranks)
        assert cost > 0
        se = (bern.var() / len(x) + v.var() / len(x)).sqrt().item()
        assert abs(v.mean().item() - bern.mean().item()) < 4 * se


def test_exact_logit_ties_follow_argmax():
    # all logits 0 (every neuron off): torch.argmax picks position 0
    from algzoo_interp.lines import _correct_interval
    L = th.zeros(3, 3, dtype=th.float64)
    lo, hi = _correct_interval(L, th.zeros_like(L), th.zeros(3, dtype=th.float64), th.tensor([0, 1, 2]))
    assert (hi > lo).tolist() == [True, False, False]
