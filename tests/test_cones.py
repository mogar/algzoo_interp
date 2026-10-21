import math

import pytest
import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import load_2nd_argmax, randn_inputs
from algzoo_interp.cones import area, clip, enumerate_cones, exact_accuracy, octants


def test_octants_tile_sphere():
    assert sum(area(p) for p in octants()) == pytest.approx(4 * math.pi)
    assert area(octants()[0]) == pytest.approx(math.pi / 2)


def test_clip_halves_octant():
    # The plane x0 = x1 cuts the positive octant into two equal halves
    n = th.tensor([1.0, -1.0, 0.0], dtype=th.float64)
    half = clip(octants()[0], n)
    other = clip(octants()[0], -n)
    assert area(half) == pytest.approx(math.pi / 4)
    assert area(half) + area(other) == pytest.approx(math.pi / 2)


def test_exact_accuracy_matches_sampling():
    model = load_2nd_argmax(4, 3).double()
    cones = enumerate_cones(model)
    assert sum(c.prob for c in cones) == pytest.approx(1.0, abs=1e-9)
    assert sum(sum(c.confusion.values()) for c in cones) == pytest.approx(1.0, abs=1e-9)
    x = randn_inputs(2**18, 3, dtype=th.float64)
    sampled = (model(x).argmax(-1) == task_2nd_argmax(x)).double().mean().item()
    stderr = (sampled * (1 - sampled) / len(x)) ** 0.5
    assert abs(exact_accuracy(cones) - sampled) < 4 * stderr
