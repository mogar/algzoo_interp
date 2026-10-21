import pytest
import torch as th
from alg_zoo import handcrafted_2nd_argmax

from algzoo_interp import randn_inputs, run_with_cache
from algzoo_interp.dictionary import ValueFitter, build_dictionary
from algzoo_interp.features import events, lagged, max0_excluding_lag, running_top2, steps_since
from algzoo_interp.state import cellwise_r2, ideal_state


def test_lagged():
    x = th.arange(1.0, 6.0)[None]
    assert lagged(x, 2).tolist() == [[0, 0, 1, 2, 3]]


def test_running_top2():
    x = th.tensor([[1.0, 3.0, 2.0, 5.0]])
    top1, top2 = running_top2(x)
    assert top1.tolist() == [[1, 3, 3, 5]]
    assert top2[0, 1:].tolist() == [1, 2, 3]
    assert th.isinf(top2[0, 0])


def test_events():
    x = th.tensor([[-1.0, 2.0, 1.0, 3.0, 0.5]])
    ev = events(x)
    assert ev["new_max"].tolist() == [[True, True, False, True, False]]
    assert ev["new_max_pos"].tolist() == [[False, True, False, True, False]]
    assert ev["new_top2"].tolist() == [[True, True, True, True, False]]


def test_steps_since():
    e = th.tensor([[False, True, False, False, True]])
    assert steps_since(e).tolist() == [[1, 0, 1, 2, 0]]


def test_max0_excluding_lag():
    x = th.tensor([[3.0, -1.0, 1.0]])
    # at t=2, excluding x_{t-2} = x_0 leaves max(0, -1, 1) = 1
    assert max0_excluding_lag(x, 2)[0, 2].item() == 1.0
    assert max0_excluding_lag(x, 1)[0, 2].item() == 3.0


def test_value_fitter_recovers_affine_relation():
    x = randn_inputs(4096, 5)
    h = (2.0 * x + 0.5)[:, :, None]
    cache = type("C", (), {"h": h})()
    f = ValueFitter(cache, {"x": x, "noise": randn_inputs(4096, 5, seed=1)}, steps=[1, 2, 3, 4])
    coef, r2 = f.fit(0, [0])
    assert r2 == pytest.approx(1.0)
    assert coef.tolist() == pytest.approx([2.0, 0.5], abs=1e-6)
    names, _, r2_greedy = f.greedy(0, k=2)
    assert names == ["x"] and r2_greedy == pytest.approx(1.0)


def test_dictionary_recovers_handcrafted_neurons():
    model = handcrafted_2nd_argmax(10)
    with th.no_grad():
        cache = run_with_cache(model, randn_inputs(2**13, 10))
    entries, _, _ = build_dictionary(cache)
    expected = {9 - k: f"relu(x_{{t-{k}}})" for k in range(10)}
    expected.update({18 - (k - 1): f"max0_excl(t-{k})-x_t" for k in range(1, 10)})
    expected.update({19: "max0-max0_prev", 20: "max0_prev", 21: "max0-x_t"})
    for n, feature in expected.items():
        assert entries[n].best_value == feature, n
        assert entries[n].value_r2 > 0.999, n


def test_ideal_state_and_cellwise_r2():
    x = th.tensor([[1.0, 3.0, 2.0, 0.5]])
    st = ideal_state(x)
    assert st["lag_max"].tolist() == [[0, 0, 1, 2]]
    assert st["lag_2nd"].tolist() == [[0, 1, 0, 1]]
    # A target that is exactly affine within each cell is fit perfectly
    gen = th.Generator().manual_seed(0)
    cells = th.randint(0, 5, (4096,), generator=gen)
    f = th.randn(4096, generator=gen)
    y = (cells.float() * f + cells.float() ** 2)[:, None]
    assert cellwise_r2(y, cells, [f]).item() == pytest.approx(1.0)
