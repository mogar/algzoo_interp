import torch as th

from algzoo_interp import load_2nd_argmax, randn_inputs
from algzoo_interp.estimation import errors_on, forward_macs, noisy_family, perturb, score, true_error


def test_forward_macs():
    model = load_2nd_argmax(32, 10)
    assert forward_macs(model) == 10 * (32 + 32 * 32) + 32 * 10


def test_perturb_is_relative_and_reproducible():
    model = load_2nd_argmax(4, 3)
    a, b = perturb(model, 0.01, seed=1), perturb(model, 0.01, seed=1)
    for p, q, r in zip(model.double().parameters(), a.parameters(), b.parameters()):
        assert th.equal(q, r)
        assert ((q - p).abs() <= 0.1 * p.abs() + 1e-15).all()  # within 10 sigma of relative noise
        assert th.equal(q == 0, p == 0)  # zero weights stay zero
    assert not th.equal(perturb(model, 0.01, seed=2).linear.weight, a.linear.weight)


def test_noisy_family_starts_with_the_model():
    model = load_2nd_argmax(4, 3)
    family = noisy_family(model, [0.001, 0.003], n_per=2)
    assert [m.key for m in family][:2] == ["sigma=0/i=0", "sigma=0.001/i=0"]
    assert len(family) == 5


def test_true_error_matches_direct_count():
    model = load_2nd_argmax(4, 3).double()
    p, var = true_error(model, 3000, seed=5, batch=1000)
    direct = th.cat([errors_on(model, randn_inputs(1000, 3, seed=5 + b, dtype=th.float64)) for b in range(3)])
    assert abs(p - direct.mean().item()) < 1e-12
    assert abs(var - p * (1 - p) / 3000) < 1e-15


def test_score_subtracts_truth_variance():
    mse, floor = score([0.1, 0.3], [0.2, 0.2], [0.001, 0.003])
    assert abs(floor - 0.002) < 1e-15
    assert abs(mse - (0.01 - 0.002)) < 1e-15
