import pytest
import torch as th
from alg_zoo import handcrafted_2nd_argmax

from algzoo_interp import accuracy, load_2nd_argmax, load_example_2nd_argmax, randn_inputs


@pytest.mark.parametrize("hidden_size,seq_len", [(2, 2), (16, 10), (32, 10)])
def test_load_2nd_argmax(hidden_size, seq_len):
    model = load_2nd_argmax(hidden_size, seq_len)
    assert model.hidden_size == hidden_size
    assert model.seq_len == seq_len
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == hidden_size + hidden_size**2 + seq_len * hidden_size


def test_32x10_beats_chance():
    model = load_2nd_argmax(32, 10)
    assert accuracy(model, randn_inputs(2**14, 10)) > 0.5


def test_example_model():
    model = load_example_2nd_argmax()
    assert (model.hidden_size, model.seq_len) == (16, 10)
    # The blog post's checkpoint name says 94% accuracy
    assert accuracy(model, randn_inputs(2**16, 10)) == pytest.approx(0.94, abs=0.02)


@pytest.mark.parametrize("seq_len", [2, 10])
def test_handcrafted_accuracy(seq_len):
    model = handcrafted_2nd_argmax(seq_len)
    assert accuracy(model, randn_inputs(2**14, seq_len)) > 0.99


def test_scale_invariance():
    # No biases + ReLU: f(c x) = c f(x) for c > 0
    model = load_2nd_argmax(8, 5).double()
    x = randn_inputs(1024, 5, dtype=th.float64)
    with th.no_grad():
        th.testing.assert_close(model(3.0 * x), 3.0 * model(x))
