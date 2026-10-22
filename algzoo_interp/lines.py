"""Exact error of a ReLU RNN along lines through input space.

Along a line x(s) = y + s u the model is piecewise linear in s: within one
activation pattern every pre-activation, hidden state and logit is affine in
s. `line_error_mass` walks a batch of lines from breakpoint to breakpoint
(where some pre-activation crosses zero), and within each piece finds exactly
where the true answer's logit is the largest. It returns the N(0, 1) mass,
∫ φ(s) 1[model wrong at x(s)] ds, over an interval of s.

The true answer along the line is given by a tie at s = 0: position `pos_pos`
for s > 0 and `pos_neg` for s < 0. Callers choose lines on which that holds
(e.g. u = (e_j - e_k)/√2 while j and k stay the 2nd and 3rd largest).
`tie_lines` builds such lines, and `cell_interval` finds the part of a line
inside one of the cells of algzoo_interp.order_stats.
"""

import torch as th

from .hooked import get_weights

TOL = 1e-12


def _Phi(x):
    return th.special.ndtr(x)


def _run(W_ih, W_hh, W_out, y, u, s):
    """Forward pass at x = y + s u with forward-mode derivatives in s.

    Returns logits and their slopes, every pre-activation and its slope, and the
    masks used. A pre-activation that is exactly zero takes its on/off state from
    the sign of its slope, so the pattern is the one just to the right of s.
    """
    B, T = y.shape
    H = W_hh.shape[0]
    h = th.zeros(B, H, dtype=y.dtype)
    dh = th.zeros_like(h)
    pres, dpres = [], []
    x = y + s[:, None] * u
    for t in range(T):
        pre = x[:, t : t + 1] * W_ih + h @ W_hh.T
        dpre = u[:, t : t + 1] * W_ih + dh @ W_hh.T
        on = (pre > TOL) | ((pre.abs() <= TOL) & (dpre > 0))
        h, dh = pre * on, dpre * on
        pres.append(pre)
        dpres.append(dpre)
    return h @ W_out.T, dh @ W_out.T, th.stack(pres, 1), th.stack(dpres, 1)


def _correct_interval(L, dL, s0, truth):
    """The interval of s (in absolute terms) on which logit `truth` is the largest, for affine logits."""
    c = L.gather(1, truth[:, None]) - L  # (L_truth - L_q)(s0 + r) = c + d r >= 0
    d = dL.gather(1, truth[:, None]) - dL
    inf = th.full_like(c, float("inf"))
    lo = th.where(d > 0, -c / d.where(d != 0, th.ones_like(d)), -inf)
    hi = th.where(d < 0, -c / d.where(d != 0, th.ones_like(d)), inf)
    # d == 0: always correct if c > 0, never if c < 0. On an exact tie (c == 0, e.g.
    # every neuron off, so all logits are 0) torch.argmax picks the lowest index,
    # so the truth loses to any lower q.
    q = th.arange(L.shape[1])[None, :]
    never = (d == 0) & ((c < 0) | ((c == 0) & (q < truth[:, None])))
    lo = th.where(never, inf, lo)
    return s0 + lo.amax(1), s0 + hi.amin(1)


def _phi_mass(a, b):
    """Φ(b) - Φ(a), through the upper tail when a > 0 to keep precision for narrow intervals."""
    return th.where(a > 0, _Phi(-a) - _Phi(-b), _Phi(b) - _Phi(a)).clamp(min=0)


def line_error_mass(model, y, u, a, b, pos_pos, pos_neg, max_iter=10_000):
    """∫_a^b φ(s) 1[model wrong at y + s u] ds for a batch of lines, exactly.

    y, u: [B, T]; a, b: [B] with a <= b; pos_pos / pos_neg: [B] true answer for s > 0 / s < 0.
    Returns (mass [B], number of forward-with-derivative passes per line [B]).
    """
    w = get_weights(model)
    W_ih, W_hh, W_out = (m.detach().to(y.dtype) for m in (w.W_ih, w.W_hh, w.W_out))
    W_ih = W_ih.reshape(1, -1)
    B = len(y)
    s = a.clone()
    mass = th.zeros(B, dtype=y.dtype)
    passes = th.zeros(B, dtype=th.long)
    active = s < b
    for _ in range(max_iter):
        if not active.any():
            return mass, passes
        idx = active.nonzero()[:, 0]
        L, dL, pre, dpre = _run(W_ih, W_hh, W_out, y[idx], u[idx], s[idx])
        passes[idx] += 1
        # next breakpoint: the first pre-activation crossing zero to the right of s
        r = -pre / dpre.where(dpre != 0, th.ones_like(dpre))
        r = th.where((dpre != 0) & (r > TOL), r, th.full_like(r, float("inf")))
        end = th.minimum(s[idx] + r.flatten(1).amin(1), b[idx])
        start = s[idx]
        # the truth flips at s = 0: split the piece there
        for lo_cut, hi_cut, truth in ((start, th.minimum(end, th.zeros_like(end)), pos_neg[idx]),
                                      (th.maximum(start, th.zeros_like(start)), end, pos_pos[idx])):
            seg = hi_cut > lo_cut
            ok_lo, ok_hi = _correct_interval(L, dL, start, truth)
            c_lo, c_hi = th.maximum(ok_lo, lo_cut), th.minimum(ok_hi, hi_cut)
            correct = th.where(c_hi > c_lo, _phi_mass(c_lo, c_hi), th.zeros_like(c_lo))
            mass[idx] += th.where(seg, _phi_mass(lo_cut, hi_cut) - correct, th.zeros_like(correct))
        s[idx] = end
        active = s < b
    raise RuntimeError("line walk did not finish; raise max_iter")


def tie_lines(x, ranks):
    """For each input, the line through it along the tie between the inputs at `ranks` (a, a + 1).

    With j, k the positions at ranks a and a + 1 (0 = largest) and
    u = (e_j - e_k) / √2, x = y + s u, where s = u·x ~ N(0, 1) is independent of y.
    Along the line, j and k keep ranks a and a + 1 exactly while |s| < w, and
    the tie x_j = x_k is at s = 0 (s > 0 at x itself). Returns
    (y, u, s, w, mid, others, j, k): mid = y_j = y_k, and `others` are the other
    inputs, largest first.
    """
    a = ranks[0]
    assert ranks[1] == a + 1
    B = len(x)
    rows = th.arange(B)
    order = x.argsort(-1, descending=True)
    j, k = order[:, a], order[:, a + 1]
    u = th.zeros_like(x)
    u[rows, j], u[rows, k] = 2**-0.5, -(2**-0.5)
    s = (x * u).sum(1)
    y = x - s[:, None] * u
    mid = y[rows, j]
    masked = x.clone()
    masked[rows, j] = masked[rows, k] = -float("inf")
    others = masked.sort(-1, descending=True).values[:, : x.shape[1] - 2]
    inf = th.full_like(mid, float("inf"))
    gap_above = others[:, a - 1] - mid if a > 0 else inf
    gap_below = mid - others[:, a] if a < x.shape[1] - 2 else inf
    w = 2**0.5 * th.minimum(gap_above, gap_below)
    return y, u, s, w, mid, others, j, k


def _halfline(alpha, beta, lo, hi):
    """Intersect [lo, hi] with {t : alpha + beta t >= 0}."""
    safe = beta.where(beta != 0, th.ones_like(beta))
    root = -alpha / safe
    lo = th.where(beta > 0, th.maximum(lo, root), lo)
    hi = th.where(beta < 0, th.minimum(hi, root), hi)
    empty = (beta == 0) & (alpha < 0)
    return lo, th.where(empty, th.full_like(hi, -float("inf")), hi)


def cell_interval(cell, ranks, mid, others, w):
    """The range [t_lo, t_hi] of |s| for which the line stays in a "pos" cell (algzoo_interp.order_stats).

    At |s| = t the pair is at mid ± t/√2. Every band condition lo <= (A - B)/A < hi
    with A > 0 is linear in t: (1 - lo) A - B >= 0 and B - (1 - hi) A > 0. So is
    v2 >= 0. Intersecting the half-lines with [0, w) gives the interval.
    """
    r = 2**-0.5
    up = (mid, th.full_like(mid, r))  # (value at t = 0, slope in t)
    down = (mid, th.full_like(mid, -r))
    o = (others[:, 0], th.zeros_like(mid))
    v1, v2, v3 = (up, down, o) if ranks == (0, 1) else (o, up, down)
    lo, hi = th.zeros_like(mid), w.clone()

    def lin(c1, a, c2, b):  # c1 * a + c2 * b, for (value, slope) pairs
        return c1 * a[0] + c2 * b[0], c1 * a[1] + c2 * b[1]

    lo, hi = _halfline(*v2, lo, hi)  # v2 >= 0
    for (A, Bv), (g_lo, g_hi) in (((v1, v2), cell.g12), ((v2, v3), cell.g23)):
        lo, hi = _halfline(*lin(1 - g_lo, A, -1, Bv), lo, hi)
        if g_hi != float("inf"):
            lo, hi = _halfline(*lin(1, Bv, -(1 - g_hi), A), lo, hi)
    return lo, hi
