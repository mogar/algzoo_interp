import pytest
import torch as th
from alg_zoo import handcrafted_2nd_argmax
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import (
    ablate,
    ablate_edges,
    accuracy,
    accuracy_by_top2_gap,
    accuracy_by_top2_positions,
    count_patterns,
    linear_map,
    load_2nd_argmax,
    logit_contributions,
    patch,
    randn_inputs,
    run_with_cache,
    sorted_inputs,
    vary_position,
    with_top2_gap,
    with_top2_positions,
)


@pytest.fixture(scope="module")
def model():
    return load_2nd_argmax(16, 10).double()


@pytest.fixture(scope="module")
def x():
    return randn_inputs(4096, 10, dtype=th.float64)


def test_cache_matches_model(model, x):
    cache = run_with_cache(model, x)
    with th.no_grad():
        th.testing.assert_close(cache.logits, model(x))
    assert cache.h.shape == cache.pre.shape == (4096, 10, 16)
    th.testing.assert_close(cache.h, th.relu(cache.pre))


def test_cache_time_indexing_on_handcrafted():
    # Handcrafted 10-step model: neuron 9 is relu(x_t), neurons 0-8 are a shift
    # register, so neuron 9 - k at step t is relu(x_{t-k})
    model = handcrafted_2nd_argmax(10)
    x = randn_inputs(256, 10)
    h = run_with_cache(model, x).h
    for t in range(10):
        for k in range(min(t + 1, 10)):
            th.testing.assert_close(h[:, t, 9 - k], th.relu(x[:, t - k]))


def test_logit_contributions_sum(model, x):
    cache = run_with_cache(model, x)
    th.testing.assert_close(logit_contributions(model, cache).sum(-1), cache.logits)


def test_zero_ablate_everything(model, x):
    cache = ablate(model, x, mode="zero")
    assert (cache.logits == 0).all()


def test_ablate_nothing_changes_nothing(model, x):
    cache = ablate(model, x, neurons=[], mode="mean")
    th.testing.assert_close(cache.logits, run_with_cache(model, x).logits)


def test_mean_ablation_uses_step_means(model, x):
    clean = run_with_cache(model, x)
    cache = ablate(model, x, neurons=[3], steps=[4], mode="mean")
    th.testing.assert_close(cache.h[:, 4, 3], clean.h[:, 4, 3].mean().expand(len(x)))
    th.testing.assert_close(cache.h[:, :4], clean.h[:, :4])


def test_resample_ablation_everything_is_permutation(model, x):
    cache = ablate(model, x, mode="resample", seed=1)
    perm = th.randperm(len(x), generator=th.Generator().manual_seed(1))
    th.testing.assert_close(cache.logits, run_with_cache(model, x[perm]).logits)


def test_patch_last_step_copies_source(model, x):
    x_src = randn_inputs(4096, 10, seed=1, dtype=th.float64)
    cache = patch(model, x, x_src, steps=[9])
    th.testing.assert_close(cache.logits, run_with_cache(model, x_src).logits)


def test_linear_map_is_exact(model, x):
    cache = run_with_cache(model, x[:64])
    for i in range(64):
        M = linear_map(model, cache.mask[i])
        th.testing.assert_close(M @ x[i], cache.logits[i])


def test_count_patterns(model, x):
    cache = run_with_cache(model, x)
    pc = count_patterns(cache)
    assert pc.counts.sum() == len(x)
    assert (pc.counts[:-1] >= pc.counts[1:]).all()
    assert pc.coverage[-1] == pytest.approx(1.0)
    # inverse points each input at its own pattern
    th.testing.assert_close(pc.patterns[pc.inverse], cache.mask.cpu())


def test_with_top2_positions():
    x = with_top2_positions(2048, 6, argmax_pos=4, second_pos=1)
    order = x.argsort(-1, descending=True)
    assert (order[:, 0] == 4).all() and (order[:, 1] == 1).all()
    # values are a permutation of the underlying randn draw
    base = randn_inputs(2048, 6)
    th.testing.assert_close(x.sort(-1).values, base.sort(-1).values)


def test_with_top2_gap():
    x = with_top2_gap(2048, 6, gap=0.01)
    top2 = x.topk(2, -1).values
    th.testing.assert_close(top2[:, 0] - top2[:, 1], th.full((2048,), 0.01), atol=1e-6, rtol=0)


def test_sorted_and_vary_position():
    x = sorted_inputs(100, 5)
    assert (task_2nd_argmax(x) == 3).all()
    swept = vary_position(x[0], 2, th.linspace(-1, 1, 7))
    assert swept.shape == (7, 5)
    th.testing.assert_close(swept[:, 2], th.linspace(-1, 1, 7))
    assert (swept[:, [0, 1, 3, 4]] == x[0, [0, 1, 3, 4]]).all()


def test_accuracy_breakdowns_consistent(model, x):
    acc, counts = accuracy_by_top2_positions(model, x)
    assert counts.sum() == len(x)
    assert counts.diagonal().sum() == 0
    overall = (acc.nan_to_num() * counts).sum() / counts.sum()
    assert overall.item() == pytest.approx(accuracy(model, x))

    edges = th.tensor([0.0, 0.1, 0.5, float("inf")], dtype=th.float64)
    acc_gap, counts_gap = accuracy_by_top2_gap(model, x, edges)
    assert counts_gap.sum() == len(x)
    assert ((acc_gap * counts_gap).sum() / len(x)).item() == pytest.approx(accuracy(model, x))


def test_ablate_edges(model, x):
    edited = ablate_edges(model, hh=[(3, 7)], out=[(9, 7)])
    assert edited.rnn.weight_hh_l0[3, 7] == 0 and edited.linear.weight[9, 7] == 0
    assert model.rnn.weight_hh_l0[3, 7] != 0, "original model must be untouched"
    # Removing a readout edge changes logit 9 by exactly that edge's contribution
    clean = run_with_cache(model, x)
    cut = run_with_cache(ablate_edges(model, out=[(9, 7)]), x)
    th.testing.assert_close(clean.logits[:, 9] - cut.logits[:, 9], model.linear.weight[9, 7] * clean.h[:, -1, 7])
