"""Where the near-tie errors come from, measured exactly (docs/PLAN.md step 12).

1. 2nd-3rd tie lines (algzoo_interp.lines.tie_lines). For each random
   input, the exact error along the line through it, within the relative
   near-tie window |s| < 0.03 |mid| / √2, divided by P(|s| < w). Its mean over
   inputs is the near-tie part of the error rate. It is broken down by:
   - when the max (M) arrives relative to the earlier (A) and later (B)
     element of the pair, and whether B is the last input;
   - the signed offset of the model's switch point from the tie, toward A
     or B, relative to the pair's value;
   - how much of the per-line error the positions (M, A, B) explain (R² of
     group means), with and without a bin of the max's relative gap.
2. First-order prediction. One forward pass with derivatives at the tie
   point gives the switch point s* = -ΔL / ΔL' of the pair's logit
   difference. It is compared with the exact walk.
3. How predictable the offset is. Using the first-order switch point on
   2^19 inputs, it measures how much of the offset's variance three
   descriptions explain, each fitted separately within every (M, A, B)
   position cell: the cell's mean; affine in max/mid, 4th/mid and
   last/mid; and linear in all ten inputs over mid (held out on half the
   inputs).
4. The last input integrated out. The exact error rate given x_0..x_8,
   integrated over x_9 in three pieces with constant true answers (below the
   running 2nd, between, above the max). It reports how much variance is left
   compared with 0/1 sampling.

Writes results/<model>/ties/summary.json.

Example:
    python scripts/tie_offsets.py --hidden 32 --seq-len 10
"""

import argparse
import json

import torch as th

from algzoo_interp import get_weights, randn_inputs
from algzoo_interp.estimation import errors_on
from algzoo_interp.lines import _run, line_error_mass, tie_lines
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.paths import RESULTS_DIR

GAMMA = 0.03


def group_r2(key, val):
    _, inv = key.unique(return_inverse=True)
    mean = th.zeros(int(inv.max()) + 1, dtype=val.dtype).index_add_(0, inv, val) / th.bincount(inv).to(val.dtype)
    return (1 - ((val - mean[inv]) ** 2).mean() / val.var()).item()


def tie_lines_section(model, n, seed):
    T = model.seq_len
    x = randn_inputs(n, T, seed=seed, dtype=th.float64)
    y, u, s, w, mid, others, j, k = tie_lines(x, (1, 2))
    eps = th.minimum(GAMMA * mid.abs() / 2**0.5, w)
    zero = th.zeros_like(eps)
    m_pos, _ = line_error_mass(model, y, u, zero, eps, j, k)  # truth j for s > 0
    m_neg, _ = line_error_mass(model, y, u, -eps, zero, j, k)
    Q = th.special.ndtr(w) - th.special.ndtr(-w)
    f = (m_pos + m_neg) / Q
    pM = x.argmax(1)
    A, B = th.minimum(j, k), th.maximum(j, k)

    # switch-point offsets: the wrong interval next to the tie, as a relative gap
    nd = th.special.ndtri
    off_pos = nd((0.5 + m_pos).clamp(max=1 - 1e-16)).abs() * 2**0.5 / mid.abs()
    off_neg = nd((0.5 + m_neg).clamp(max=1 - 1e-16)).abs() * 2**0.5 / mid.abs()
    j_is_B = j == B
    # wrong where the truth is A means the model leans toward B, and vice versa
    lean_B = th.where(j_is_B, off_neg, off_pos) - th.where(j_is_B, off_pos, off_neg)

    cases = {"M first (M < A < B)": pM < A, "M between (A < M < B)": (pM > A) & (pM < B),
             "M last (A < B < M)": pM > B}
    rows = []
    for name, c in cases.items():
        for last in (False, True):
            g = c & ((B == T - 1) if last else (B != T - 1)) & (mid > 0)
            if not g.any():
                continue
            q = lean_B[g].quantile(th.tensor([0.25, 0.5, 0.75], dtype=th.float64)).tolist()
            rows.append({"case": name, "B_is_last": last, "prob": g.double().mean().item(),
                         "share_of_tie_error": (f[g].sum() / f.sum()).item(), "mean": f[g].mean().item(),
                         "lean_toward_B_quartiles": q})

    mx = x.max(1).values
    g12_bin = th.bucketize(th.log10(((mx - mid) / mx).clamp(min=1e-6)),
                           th.tensor([-3, -2, -1.5, -1, -0.5], dtype=th.float64))
    triple = pM * T * T + A * T + B

    # first-order prediction at the tie point
    wts = get_weights(model)
    W_ih, W_hh, W_out = (m.detach().double() for m in (wts.W_ih, wts.W_hh, wts.W_out))
    L, dL, _, _ = _run(W_ih.reshape(1, -1), W_hh, W_out, y, u, zero)
    rows_idx = th.arange(n)
    D, dD = L[rows_idx, j] - L[rows_idx, k], dL[rows_idx, j] - dL[rows_idx, k]
    s_star = (-D / dD).clamp(-eps, eps)
    f_lin = (th.special.ndtr(s_star) - 0.5).abs() / Q
    top2 = L.topk(2, dim=1).indices
    pair_on_top = ((top2 == j[:, None]) | (top2 == k[:, None])).all(1)
    return {
        "n": n, "gamma": GAMMA, "tie_error_mean": f.mean().item(), "per_line_sd_over_mean": (f.std() / f.mean()).item(),
        "by_case": rows,
        "r2_positions": group_r2(triple, f), "r2_positions_and_g12_bin": group_r2(triple * 10 + g12_bin, f),
        "n_groups_positions_and_g12_bin": int(len((triple * 10 + g12_bin).unique())),
        "first_order": {"mean": f_lin.mean().item(), "share_of_exact_mean": (f_lin.mean() / f.mean()).item(),
                        "lines_within_0.1pct_of_exact": ((f - f_lin).abs() <= 1e-3 * f).double().mean().item(),
                        "corr_with_exact": th.corrcoef(th.stack([f, f_lin]))[0, 1].item(),
                        "pair_are_top_two_logits_at_tie": pair_on_top.double().mean().item()},
    }


def offset_predictability(model, n, seed):
    T = model.seq_len
    x = randn_inputs(n, T, seed=seed, dtype=th.float64)
    y, u, s, w, mid, others, j, k = tie_lines(x, (1, 2))
    wts = get_weights(model)
    W_ih, W_hh, W_out = (m.detach().double() for m in (wts.W_ih, wts.W_hh, wts.W_out))
    L, dL, _, _ = _run(W_ih.reshape(1, -1), W_hh, W_out, y, u, th.zeros(n, dtype=th.float64))
    r = th.arange(n)
    A, B = th.minimum(j, k), th.maximum(j, k)
    lean = (-(L[r, j] - L[r, k]) / (dL[r, j] - dL[r, k])) * 2**0.5 / mid
    lean = th.where(j == B, lean, -lean)  # relative offset of the switch point, positive toward A
    ok = (mid > 0.2) & (lean.abs() < 0.05)
    key = x.argmax(1) * T * T + A * T + B
    one = th.ones(n, 1, dtype=th.float64)
    ratios = th.cat([one, th.stack([x.max(1).values, others[:, 1], y[:, -1]], 1) / mid[:, None]], 1)
    linear = th.cat([one, y / mid[:, None]], 1)
    half = th.rand(n, generator=th.Generator().manual_seed(seed)) < 0.5
    out = {"n_lines": int(ok.sum()), "offset_sd": lean[ok].std().item()}
    for name, feats, held_out in (("cell_mean", one, False), ("affine_in_3_ratios", ratios, False),
                                  ("linear_in_all_inputs_held_out", linear, True)):
        fit_mask, test_mask = (ok & half, ok & ~half) if held_out else (ok, ok)
        base = lean[test_mask].mean()
        res = tot = 0.0
        for kk in key[ok].unique():
            tr, te = fit_mask & (key == kk), test_mask & (key == kk)
            if tr.sum() < 200 or te.sum() < 50:
                continue
            beta = th.linalg.lstsq(feats[tr], lean[tr][:, None]).solution
            res += ((lean[te] - (feats[te] @ beta)[:, 0]) ** 2).sum().item()
            tot += ((lean[te] - base) ** 2).sum().item()
        out[f"r2_{name}"] = 1 - res / tot if tot > 0 else None  # None: no cell had enough lines
    return out


def last_input_section(model, n, seed):
    T = model.seq_len
    x = randn_inputs(n, T, seed=seed, dtype=th.float64)
    y = x.clone()
    y[:, -1] = 0
    u = th.zeros_like(x)
    u[:, -1] = 1
    order = x[:, :-1].argsort(-1, descending=True)
    r1, r2 = x.gather(1, order[:, :1])[:, 0], x.gather(1, order[:, 1:2])[:, 0]
    p1, p2, last = order[:, 0], order[:, 1], th.full_like(order[:, 0], T - 1)
    lo, hi = th.full_like(r1, -12.0), th.full_like(r1, 12.0)
    below, _ = line_error_mass(model, y, u, lo, r2, p2, p2)
    between, _ = line_error_mass(model, y, u, r2, r1, last, last)
    above, _ = line_error_mass(model, y, u, r1, hi, p1, p1)
    P = below + between + above
    e = errors_on(model, x)
    q = P.quantile(th.tensor([0.5, 0.9, 0.99, 0.999], dtype=th.float64)).tolist()
    return {"n": n, "mean": P.mean().item(), "mean_se": (P.std() / n**0.5).item(),
            "plain_mean": e.mean().item(), "plain_se": (e.std() / n**0.5).item(),
            "variance_reduction": (e.var() / P.var()).item(),
            "by_piece": {"last input below the running 2nd": below.mean().item(),
                         "last input is the 2nd": between.mean().item(),
                         "last input is the max": above.mean().item()},
            "quantiles_50_90_99_999": q}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**18)
    parser.add_argument("--n-last", type=int, default=2**16)
    args = parser.parse_args()
    model, name = load_from_args(args)
    model = model.double().eval()
    out = RESULTS_DIR / name / "ties"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Tie offsets for {name} -> {out}")
    S = {"model": name, "tie_lines": tie_lines_section(model, args.n, seed=31),
         "offset_predictability": offset_predictability(model, 2 * args.n, seed=51),
         "last_input": last_input_section(model, args.n_last, seed=41)}
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    t, li = S["tie_lines"], S["last_input"]
    print(f"  tie error {t['tie_error_mean']:.5f}; R² positions {t['r2_positions']:.3f}, "
          f"+g12 bin {t['r2_positions_and_g12_bin']:.3f}; first order exact on "
          f"{t['first_order']['lines_within_0.1pct_of_exact']:.2f} of lines, "
          f"{t['first_order']['share_of_exact_mean']:.2f} of the mean")
    for r in t["by_case"]:
        print(f"  {r['case']:24s} B last={r['B_is_last']!s:5s} P {r['prob']:.3f} share {r['share_of_tie_error']:.3f} "
              f"mean {r['mean']:.5f} lean→B quartiles {[round(v, 5) for v in r['lean_toward_B_quartiles']]}")
    print(f"  offset predictability: {S['offset_predictability']}")
    print(f"  last input integrated: mean {li['mean']:.5f} ± {li['mean_se']:.5f} (plain {li['plain_mean']:.5f}); "
          f"variance reduction {li['variance_reduction']:.2f}x; {li['by_piece']}")


if __name__ == "__main__":
    main()
