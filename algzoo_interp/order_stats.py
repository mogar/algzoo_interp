"""Exact probabilities and conditional samplers for cells of n iid N(0, 1) inputs.

A cell is defined by the order statistics v1 > v2 > v3 (largest, 2nd, 3rd):

- "max<0": every input is negative;
- "2nd<0": exactly one input is positive;
- "pos": v2 >= 0, with the relative gaps g12 = (v1 - v2) / v1 and
  g23 = (v2 - v3) / v2 in the half-open bands [lo, hi). g12 is in [0, 1). g23
  is above 1 exactly when v3 < 0, so its last band can be [1, inf).

The probability of a "pos" cell is a 1-D integral over v2. The joint density
of the top three is n(n-1)(n-2) φ(v1) φ(v2) φ(v3) Φ(v3)^(n-3), so

    P = ∫_0^∞ n(n-1) φ(v2) [Φ(U1) - Φ(L1)] [Φ(U3)^(n-2) - Φ(L3)^(n-2)] dv2

with v1 in [L1, U1] = [v2 / (1 - g12_lo), v2 / (1 - g12_hi)] and v3 in
[L3, U3] = [v2 (1 - g23_hi), v2 (1 - g23_lo)]. The samplers draw v2 from
that integrand, which is tabulated on the quadrature grid and inverted by
interpolation. Given v2, they draw v1, v3 and the other n - 3 inputs exactly
from their truncated distributions, then shuffle the positions.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch as th

V2_MAX = 10.0  # P(v2 > 10) is below 1e-22 for n = 10
GRID = 2**16  # quadrature intervals (Simpson's rule)


@dataclass(frozen=True)
class Cell:
    kind: str  # "max<0", "2nd<0" or "pos"
    g12: tuple = (0.0, 1.0)
    g23: tuple = (0.0, float("inf"))

    @property
    def name(self):
        if self.kind != "pos":
            return self.kind
        return f"g12 [{self.g12[0]:g},{self.g12[1]:g}) g23 [{self.g23[0]:g},{self.g23[1]:g})"


def gap_cells(g12_edges, g23_edges):
    """The two negative cells plus a "pos" cell for every pair of bands. Together they partition the inputs."""
    cells = [Cell("max<0"), Cell("2nd<0")]
    for a in zip(g12_edges[:-1], g12_edges[1:]):
        for b in zip(g23_edges[:-1], g23_edges[1:]):
            cells.append(Cell("pos", a, b))
    return cells


def _Phi(x):
    return th.special.ndtr(x)


def _bounds(cell, v2):
    (a12, b12), (a23, b23) = cell.g12, cell.g23
    inf = th.full_like(v2, float("inf"))
    L1 = v2 / (1 - a12)
    U1 = inf if b12 >= 1 else v2 / (1 - b12)
    L3 = -inf if b23 == float("inf") else v2 * (1 - b23)
    U3 = v2 * (1 - a23)
    return L1, U1, L3, U3


def _density(cell, v2, n):
    """The integrand above: the density of v2 jointly with the cell, for v2 >= 0."""
    L1, U1, L3, U3 = _bounds(cell, v2)
    phi = th.exp(-v2**2 / 2) / (2 * th.pi) ** 0.5
    p1 = _Phi(-L1) - _Phi(-U1)  # Φ(U1) - Φ(L1), via upper tails for precision
    p3 = _Phi(U3) ** (n - 2) - _Phi(L3) ** (n - 2)
    return n * (n - 1) * phi * p1 * p3


def _grid():
    return th.linspace(0, V2_MAX, GRID + 1, dtype=th.float64)


def _simpson_weights():
    w = th.ones(GRID + 1, dtype=th.float64)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * (V2_MAX / GRID) / 3


def cell_prob(cell, n):
    if cell.kind == "max<0":
        return 2.0**-n
    if cell.kind == "2nd<0":
        return n * 2.0**-n
    return (_density(cell, _grid(), n) * _simpson_weights()).sum().item()


def _truncated_below(hi, size, gen):
    """N(0, 1) conditioned on x < hi."""
    u = th.rand(size, generator=gen, dtype=th.float64)
    return th.special.ndtri(u * _Phi(hi))


@lru_cache(maxsize=1024)
def _v2_table(cell, n):
    """The grid and the normalised CDF of v2 within the cell (trapezoid rule)."""
    v = _grid()
    f = _density(cell, v, n)
    cdf = th.cat([th.zeros(1, dtype=th.float64), th.cumsum((f[1:] + f[:-1]) / 2 * (v[1] - v[0]), 0)])
    return v, cdf / cdf[-1]


def sample_cell(cell, n, size, gen):
    """`size` inputs [size, n] drawn from N(0, I) conditioned on the cell."""
    if cell.kind == "max<0":
        return -th.randn(size, n, generator=gen, dtype=th.float64).abs()
    if cell.kind == "2nd<0":
        x = -th.randn(size, n, generator=gen, dtype=th.float64).abs()
        pos = th.randint(n, (size,), generator=gen)
        x[th.arange(size), pos] *= -1
        return x
    # v2 by inverting the tabulated CDF of the density on the grid
    v, cdf = _v2_table(cell, n)
    u = th.rand(size, generator=gen, dtype=th.float64)
    i = th.searchsorted(cdf, u).clamp(1, GRID)
    t = (u - cdf[i - 1]) / (cdf[i] - cdf[i - 1]).clamp(min=1e-300)
    v2 = v[i - 1] + t * (v[i] - v[i - 1])
    L1, U1, L3, U3 = _bounds(cell, v2)
    # v1: N(0, 1) on [L1, U1], through the upper tail
    u = th.rand(size, generator=gen, dtype=th.float64)
    v1 = -th.special.ndtri(_Phi(-L1) - u * (_Phi(-L1) - _Phi(-U1)))
    v1 = th.maximum(th.minimum(v1, U1), L1)
    # v3: density ∝ φ Φ^(n-3) on [L3, U3], i.e. Φ(v3)^(n-2) uniform between the ends
    u = th.rand(size, generator=gen, dtype=th.float64)
    lo, hi = _Phi(L3) ** (n - 2), _Phi(U3) ** (n - 2)
    v3 = th.special.ndtri((lo + u * (hi - lo)) ** (1 / (n - 2)))
    v3 = th.maximum(th.minimum(v3, U3), L3)
    rest = _truncated_below(v3[:, None].expand(size, n - 3), (size, n - 3), gen)
    x = th.cat([v1[:, None], v2[:, None], v3[:, None], rest], 1)
    perm = th.rand(size, n, generator=gen).argsort(1)
    return x.gather(1, perm)


def cell_index(x, cells):
    """Index into `cells` of the cell each input falls in (-1 if none)."""
    s = x.sort(-1, descending=True).values
    v1, v2, v3 = s[:, 0], s[:, 1], s[:, 2]
    g12 = (v1 - v2) / v1.clamp(min=1e-300)
    g23 = (v2 - v3) / v2.clamp(min=1e-300)
    out = th.full((len(x),), -1, dtype=th.long)
    for k, c in enumerate(cells):
        if c.kind == "max<0":
            mask = v1 < 0
        elif c.kind == "2nd<0":
            mask = (v1 >= 0) & (v2 < 0)
        else:
            mask = (v2 >= 0) & (g12 >= c.g12[0]) & (g12 < c.g12[1]) & (g23 >= c.g23[0]) & (g23 < c.g23[1])
        out[mask] = k
    return out
