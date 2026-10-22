import pytest
import torch as th

from algzoo_interp import randn_inputs
from algzoo_interp.order_stats import Cell, cell_index, cell_prob, gap_cells, sample_cell

G12 = [0, 1e-3, 0.03, 0.3, 1.0]
G23 = [0, 1e-3, 0.03, 0.3, 1.0, float("inf")]


@pytest.mark.parametrize("n", [3, 10])
def test_cell_probs_partition(n):
    cells = gap_cells(G12, G23)
    assert abs(sum(cell_prob(c, n) for c in cells) - 1) < 1e-10
    assert cell_prob(Cell("max<0"), n) == 2.0**-n
    assert cell_prob(Cell("2nd<0"), n) == n * 2.0**-n


def test_cell_probs_match_sampling():
    cells = gap_cells(G12, G23)
    P = th.tensor([cell_prob(c, 10) for c in cells])
    x = randn_inputs(2**20, 10, seed=4, dtype=th.float64)
    idx = cell_index(x, cells)
    assert (idx >= 0).all()
    freq = th.bincount(idx, minlength=len(cells)).double() / len(x)
    z = (freq - P) / (P * (1 - P) / len(x)).sqrt()
    assert z.abs().max() < 4.5


@pytest.mark.parametrize("n", [3, 10])
def test_samples_land_in_their_cell(n):
    cells = gap_cells(G12, G23)
    gen = th.Generator().manual_seed(0)
    for k, c in enumerate(cells):
        x = sample_cell(c, n, 500, gen)
        assert x.shape == (500, n)
        assert (cell_index(x, cells) == k).all(), c.name


def test_sampler_positions_are_uniform():
    gen = th.Generator().manual_seed(1)
    x = sample_cell(Cell("pos", (0.03, 0.3), (0.0, 0.03)), 10, 20000, gen)
    second = x.argsort(-1, descending=True)[:, 1]
    counts = th.bincount(second, minlength=10).double()
    assert ((counts - 2000).abs() < 4 * 2000**0.5).all()
