"""Are the errors set by the top values, or by crosstalk from the others? (docs/PLAN.md steps 12-13)

For the error-prone cells (only the max positive; all negative; a 2nd-3rd
near-tie) and a typical cell, using exact cell samplers
(algzoo_interp.order_stats):

1. Negative cell structure: the rank of the wrong prediction, and the error
   rate by the 2nd-3rd gap relative to the max and by how negative the 2nd
   is (relative to the max).
2. Resample check: keep the positions and values of the top three (or
   four), redraw every other input from N(0, 1) below them, and count how
   often the prediction changes. If the top values decided the answer, it
   would rarely change.
3. Linear crosstalk: predict the change in the logit difference between
   the true 2nd and 3rd from its gradient with respect to the redrawn
   inputs. Compare it with the actual change (R²), and the predicted sign
   flips with the actual ones.
4. Which neurons carry the flips: for redraw pairs whose prediction flips,
   greedily copy final hidden states from the redrawn input until the
   flipped prediction is reproduced.

Writes results/<model>/crosstalk/summary.json.

Example:
    python scripts/crosstalk.py --hidden 32 --seq-len 10
"""

import argparse
import json

import torch as th

from algzoo_interp import get_weights, run_with_cache
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.order_stats import Cell, sample_cell
from algzoo_interp.paths import RESULTS_DIR

CELLS = {"only the max positive": Cell("2nd<0"), "all negative": Cell("max<0"),
         "2nd-3rd near-tie (g12 >= 0.03, g23 < 0.003)": Cell("pos", (0.03, 1.0), (0.0, 0.003)),
         "typical (g12, g23 >= 0.1)": Cell("pos", (0.1, 1.0), (0.1, 1.0))}


def predict(model, x):
    with th.no_grad():
        return th.cat([model(b).argmax(-1) for b in x.split(2**16)])


def redraw_below(x, keep, gen):
    """Keep the `keep` largest inputs; redraw the rest from N(0, 1) truncated below the smallest kept."""
    order = x.argsort(-1, descending=True)
    rows = th.arange(len(x))
    floor = x[rows, order[:, keep - 1]]
    other = th.ones_like(x, dtype=th.bool)
    for r in range(keep):
        other[rows, order[:, r]] = False
    u = th.rand(x.shape, generator=gen, dtype=th.float64)
    new = th.special.ndtri(u * th.special.ndtr(floor)[:, None])
    out = x.clone()
    out[other] = new[other]
    return out, other


def negative_structure(model, n, gen):
    x = sample_cell(Cell("2nd<0"), model.seq_len, n, gen)
    pred = predict(model, x)
    order = x.argsort(-1, descending=True)
    s = x.gather(1, order)
    wrong = pred != order[:, 1]
    rank = (order == pred[:, None]).double().argmax(-1)
    gap = (s[:, 1] - s[:, 2]) / s[:, 0]
    depth = -s[:, 1] / s[:, 0]
    gap_edges, depth_edges = [0.003, 0.01, 0.03, 0.1, 0.3], [0.03, 0.1, 0.3, 1.0]
    gi = th.bucketize(gap, th.tensor(gap_edges, dtype=th.float64))
    di = th.bucketize(depth, th.tensor(depth_edges, dtype=th.float64))
    table = [[wrong[(gi == a) & (di == b)].double().mean().item() if ((gi == a) & (di == b)).sum() > 200 else None
              for b in range(len(depth_edges) + 1)] for a in range(len(gap_edges) + 1)]
    return {"error": wrong.double().mean().item(),
            "rank_of_wrong_prediction": th.bincount(rank[wrong], minlength=model.seq_len).tolist(),
            "gap_over_max_edges": gap_edges, "depth_over_max_edges": depth_edges,
            "error_by_gap_rows_depth_cols": table}


def resample_and_linear(model, cell, n, gen):
    T = model.seq_len
    x = sample_cell(cell, T, n, gen)
    p = predict(model, x)
    truth = x.argsort(-1, descending=True)[:, 1]
    out = {"error": (p != truth).double().mean().item()}
    for keep in (3, 4):
        x2, _ = redraw_below(x, keep, gen)
        p2 = predict(model, x2)
        out[f"changed_keeping_top_{keep}"] = (p2 != p).double().mean().item()
    # linear crosstalk, on a subset
    m = min(n, 2**14)
    xs = x[:m]
    order = xs.argsort(-1, descending=True)
    rows = th.arange(m)
    xg = xs.clone().requires_grad_(True)
    L = model(xg)
    D = L[rows, order[:, 1]] - L[rows, order[:, 2]]
    (grad,) = th.autograd.grad(D.sum(), xg)
    x2, other = redraw_below(xs, 3, gen)
    with th.no_grad():
        L2 = model(x2)
    D, D2 = D.detach(), L2[rows, order[:, 1]] - L2[rows, order[:, 2]]
    lin = ((x2 - xs) * grad * other).sum(1)
    change = D2 - D
    out["linear_crosstalk_r2"] = (1 - ((change - lin) ** 2).mean() / change.var()).item()
    out["sign_flips_actual"] = ((D2 > 0) != (D > 0)).double().mean().item()
    out["sign_flips_linear"] = ((D + lin > 0) != (D > 0)).double().mean().item()
    return out


def flip_carriers(model, cell, n, gen, steps=12):
    x = sample_cell(cell, model.seq_len, n, gen)
    x2, _ = redraw_below(x, 3, gen)
    with th.no_grad():
        c1, c2 = run_with_cache(model, x), run_with_cache(model, x2)
    p1, p2 = c1.logits.argmax(-1), c2.logits.argmax(-1)
    flip = p1 != p2
    W = get_weights(model).W_out.detach().double()
    h, h_src, target = c1.h[flip, -1].clone(), c2.h[flip, -1], p2[flip]
    chosen, curve = [], []
    for _ in range(steps):
        best = None
        for i in range(h.shape[1]):
            if i in chosen:
                continue
            hh = h.clone()
            hh[:, i] = h_src[:, i]
            moved = ((hh @ W.T).argmax(-1) == target).double().mean().item()
            if best is None or moved > best[1]:
                best = (i, moved)
        chosen.append(best[0])
        h[:, best[0]] = h_src[:, best[0]]
        curve.append({"add": best[0], "reproduced": best[1]})
    return {"flipping_pairs": int(flip.sum()), "greedy": curve}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**17)
    args = parser.parse_args()
    model, name = load_from_args(args)
    model = model.double().eval()
    out = RESULTS_DIR / name / "crosstalk"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Crosstalk for {name} -> {out}")
    gen = th.Generator().manual_seed(0)
    S = {"model": name, "n": args.n, "negative_cell": negative_structure(model, args.n, gen),
         "cells": {label: resample_and_linear(model, c, args.n, gen) for label, c in CELLS.items()},
         "flip_carriers": {label: flip_carriers(model, CELLS[label], args.n // 2, gen)
                           for label in list(CELLS)[:3:2]}}
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    ng = S["negative_cell"]
    print(f"  only max positive: error {ng['error']:.3f}; rank of wrong prediction {ng['rank_of_wrong_prediction']}")
    for label, r in S["cells"].items():
        print(f"  {label:44s} error {r['error']:.4f}; changed keeping top 3 {r['changed_keeping_top_3']:.4f}, "
              f"top 4 {r['changed_keeping_top_4']:.4f}; linear R² {r['linear_crosstalk_r2']:.2f}, flips "
              f"{r['sign_flips_actual']:.3f} actual vs {r['sign_flips_linear']:.3f} linear")
    for label, r in S["flip_carriers"].items():
        print(f"  {label}: {r['flipping_pairs']} flips; greedy " +
              " ".join(f"+{c['add']}→{c['reproduced']:.2f}" for c in r["greedy"]))


if __name__ == "__main__":
    main()
