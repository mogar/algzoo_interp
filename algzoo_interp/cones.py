"""Exact cone enumeration and accuracy for seq_len-3 models.

With 3 inputs, every activation-pattern cone is a convex polygon on the unit
sphere, so its probability under x ~ N(0, I) is its area / 4π (the Gaussian
is rotationally symmetric). This module enumerates the cones exactly by
splitting the sphere along each neuron's on/off plane, step by step, then
splits each final cone by predicted and true class, giving the model's exact
accuracy without sampling. See docs/GEOMETRY.md for the background.
"""

import itertools
import math
from dataclasses import dataclass, field

import torch as th

from .hooked import get_weights

EPS_AREA = 1e-12


def _normalize(v):
    return v / v.norm()


def clip(poly, n):
    """Clip a convex spherical polygon (list of unit 3-vectors, in order) to {x : n·x >= 0}."""
    if not poly:
        return []
    d = [float(n @ v) for v in poly]
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        da, db = d[i], d[(i + 1) % len(poly)]
        if da >= 0:
            out.append(a)
        if da * db < 0:
            # Point on the arc a-b where n·x = 0, as a positive combination of a and b
            p = (da * b - db * a) * (1 if da > 0 else -1)
            out.append(_normalize(p))
    return out


def area(poly):
    """Area of a convex spherical polygon (solid angle of its cone), by triangle fan."""
    if len(poly) < 3:
        return 0.0
    total = 0.0
    a = poly[0]
    for b, c in zip(poly[1:-1], poly[2:]):
        num = abs(float(a @ th.linalg.cross(b, c)))
        den = 1 + float(a @ b) + float(b @ c) + float(c @ a)
        total += 2 * math.atan2(num, den)
    return total


def clip_all(poly, normals):
    for n in normals:
        poly = clip(poly, n)
        if len(poly) < 3:
            return []
    return poly


def octants():
    eye = th.eye(3, dtype=th.float64)
    return [[s0 * eye[0], s1 * eye[1], s2 * eye[2]] for s0, s1, s2 in itertools.product([1, -1], repeat=3)]


@dataclass
class Cone:
    pattern: th.Tensor  # [3, hidden] bool
    M: th.Tensor  # [n_out, 3] exact linear map from x to logits
    prob: float  # probability under N(0, I)
    # prob of (predicted, true) class pairs within this cone
    confusion: dict = field(default_factory=dict)

    @property
    def accuracy(self):
        correct = sum(p for (pred, true), p in self.confusion.items() if pred == true)
        return correct / self.prob if self.prob > 0 else float("nan")


def true_class_cones(seq_len=3):
    """For each position p, the cones (as lists of normals) where x_p is the median of 3."""
    eye = th.eye(seq_len, dtype=th.float64)
    out = {}
    for p in range(seq_len):
        a, b = [q for q in range(seq_len) if q != p]
        # x_a <= x_p <= x_b, or x_b <= x_p <= x_a
        out[p] = [[eye[p] - eye[a], eye[b] - eye[p]], [eye[p] - eye[b], eye[a] - eye[p]]]
    return out


def _argmax_normals(M, pred, tol=1e-12):
    """Normals of the region where logit `pred` is the argmax, or None if it is empty.

    Matches torch.argmax's tie-break (lowest index wins): where logits `pred`
    and q are identical everywhere in the cone (e.g. all neurons off, so all
    logits are 0), pred loses to any q < pred and beats any q > pred.
    """
    normals = []
    for q in range(M.shape[0]):
        if q == pred:
            continue
        n = M[pred] - M[q]
        if n.norm() < tol:
            if q < pred:
                return None
            continue
        normals.append(n)
    return normals


def enumerate_cones(model):
    """All activation-pattern cones of a seq_len-3 model, with exact probabilities and confusion."""
    w = get_weights(model)
    W_ih, W_hh, W_out = (m.detach().double().cpu() for m in (w.W_ih, w.W_hh, w.W_out))
    hidden = W_ih.shape[0]
    assert model.seq_len == 3, "exact cone enumeration is implemented for seq_len 3 only"

    # Each region: (polygon pieces, A with h_{t-1} = A x, pattern rows so far).
    # The octants are disjoint, so a cone may be made of pieces from several octants.
    regions = [([poly], th.zeros(hidden, 3, dtype=th.float64), []) for poly in octants()]
    for t in range(3):
        e_t = th.zeros(3, dtype=th.float64)
        e_t[t] = 1
        new_regions = {}
        for pieces, A, rows in regions:
            N = W_ih[:, None] * e_t[None, :] + W_hh @ A  # pre_t = N x
            for bits in itertools.product([True, False], repeat=hidden):
                signs = [N[i] if on else -N[i] for i, on in enumerate(bits)]
                kept = [q for q in (clip_all(poly, signs) for poly in pieces) if q and area(q) > EPS_AREA]
                if not kept:
                    continue
                D = th.tensor(bits, dtype=th.float64)
                key = tuple(tuple(r) for r in rows) + (bits,)
                A_t = D[:, None] * N
                if key in new_regions:
                    new_regions[key][0].extend(kept)
                else:
                    new_regions[key] = (kept, A_t, rows + [bits])
        regions = list(new_regions.values())

    truth = true_class_cones()
    cones = []
    for pieces, A, rows in regions:
        M = W_out @ A
        n_out = M.shape[0]
        cone = Cone(pattern=th.tensor(rows), M=M, prob=sum(area(q) for q in pieces) / (4 * math.pi))
        for pred in range(n_out):
            pred_normals = _argmax_normals(M, pred)
            if pred_normals is None:
                continue
            pred_pieces = [r for r in (clip_all(q, pred_normals) for q in pieces) if r]
            if not pred_pieces:
                continue
            for true, options in truth.items():
                p = sum(area(clip_all(q, normals)) for q in pred_pieces for normals in options) / (4 * math.pi)
                if p > 0:
                    cone.confusion[(pred, true)] = p
        cones.append(cone)
    cones.sort(key=lambda c: -c.prob)
    return cones


def exact_accuracy(cones):
    return sum(p for c in cones for (pred, true), p in c.confusion.items() if pred == true)
