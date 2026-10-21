"""Quantitative explanation checks for a 10-step 2nd argmax model (roadmap step 6).

The explanation under test (docs/HYPOTHESES.md): the final hidden state is a
function of the ideal online state (where the max and 2nd max are, their
values, and the current input), and the readout decodes the 2nd max's
position from it.

1. Surrogate rebuild: replace the final hidden state with a prediction from
   the ideal state alone (one affine fit per (lag of max, lag of 2nd) cell,
   fitted on one sample and evaluated on a fresh one), then apply the model's
   own readout. Three levels: positions only, + values, + current input.
2. Resample check: resample what the explanation calls irrelevant, keeping
   the true answer fixed, and see whether predictions survive:
   L1 resamples every input other than the max, the 2nd max and the last
   input (from below the 2nd max); L2 resamples all values, keeping only the
   positions of the max and 2nd max. The ceiling if errors are independent
   near-ties is about (accuracy)^2.
3. Errors: gaps between ranked values among errors vs overall, what errors
   predict, and error rate vs the relative 2nd-3rd gap.
4. A one-neuron error model: predict errors where the 3rd largest arrives
   after the max and 2nd from the new-top-2 detector's firing curve alone
   (no model run on the evaluation sample), and compare with measured.

Writes results/<model>/explain/{summary.json, errors.png}.

Example:
    python scripts/explain_32x10.py --hidden 32 --seq-len 10 --top2-neuron 16
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import randn_inputs, run_with_cache
from algzoo_interp.features import lagged, running_top2
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.plots import LINE_COLOR, savefig
from algzoo_interp.state import continuous_features, ideal_state
from algzoo_interp.paths import RESULTS_DIR

LEVELS = ["none", "values", "values+input"]
SHORTFALL_EDGES = [0, .001, .002, .003, .005, .0075, .01, .015, .02, .03, .05, 1e9]


def final_cells_and_feats(x, level):
    st = ideal_state(x)
    T = x.shape[1]
    cells = st["lag_max"][:, -1] * T + st["lag_2nd"][:, -1]
    feats = [th.ones(len(x), dtype=x.dtype)] + [f[:, -1] for f in continuous_features(x, st, level)]
    return cells, th.stack(feats, 1)


def surrogate(model, x_fit, h_fit, x_eval, level):
    """Per-cell affine prediction of the final hidden state from the ideal state."""
    T = x_fit.shape[1]
    c_fit, X_fit = final_cells_and_feats(x_fit, level)
    c_eval, X_eval = final_cells_and_feats(x_eval, level)
    k, G = X_fit.shape[1], T * T
    XtX = th.zeros(G, k, k, dtype=th.float64).index_add_(0, c_fit, X_fit[:, :, None] * X_fit[:, None, :])
    XtY = th.zeros(G, k, h_fit.shape[1], dtype=th.float64).index_add_(0, c_fit, X_fit[:, :, None] * h_fit[:, None, :])
    coef = th.linalg.lstsq(XtX + 1e-6 * th.eye(k, dtype=th.float64), XtY).solution
    return (X_eval[:, :, None] * coef[c_eval]).sum(1)


def resampled(x, gen):
    """L1 and L2 resamples of x (see module docstring). Both keep the true answer."""
    order = x.argsort(-1, descending=True)
    p1, p2 = order[:, :1], order[:, 1:2]
    v2 = x.gather(1, p2)
    u = th.rand(x.shape, generator=gen, dtype=x.dtype) * th.special.ndtr(v2)
    below = th.special.ndtri(u.clamp(min=1e-300))
    keep = th.zeros_like(x, dtype=th.bool)
    keep.scatter_(1, p1, True)
    keep.scatter_(1, p2, True)
    keep[:, -1] = True
    x1 = th.where(keep, x, below)
    ys = th.randn(x.shape, generator=gen, dtype=x.dtype).sort(-1, descending=True).values
    rest = th.ones_like(x, dtype=th.bool)
    rest.scatter_(1, p1, False)
    rest.scatter_(1, p2, False)
    x2 = th.empty_like(x)
    x2[rest] = ys[:, 2:].reshape(-1)
    x2.scatter_(1, p1, ys[:, :1])
    x2.scatter_(1, p2, ys[:, 1:2])
    return x1, x2


def detector_curve(cache, neuron):
    """P(detector on | relative shortfall of x_t below the running 2nd max), steps 2..T-1."""
    x = cache.x.cpu()
    top1, top2 = running_top2(x)
    r2 = lagged(th.where(th.isinf(top2), th.zeros_like(top2), top2), 1)[:, 2:]
    r1 = lagged(top1, 1)[:, 2:]
    xt, on = x[:, 2:], cache.mask.cpu()[:, 2:, neuron]
    sel = (xt < r1) & (r2 > 0) & (xt < r2)
    rel = ((r2 - xt) / r2)[sel]
    edges = th.tensor(SHORTFALL_EDGES, dtype=th.float64)
    b = th.bucketize(rel, edges, right=True) - 1
    counts = th.bincount(b, minlength=len(edges) - 1)
    return edges, th.bincount(b, weights=on[sel].double(), minlength=len(edges) - 1) / counts.clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--top2-neuron", type=int, default=16, help="new-top-2 detector")
    parser.add_argument("--n", type=int, default=2**20)
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.double().eval()
    out = RESULTS_DIR / name / "explain"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Explaining {name} -> {out}")
    gen = th.Generator().manual_seed(5)
    S = {"model": name, "n_inputs": args.n}

    x_fit = randn_inputs(args.n // 4, model.seq_len, seed=0, dtype=th.float64)
    x = randn_inputs(args.n, model.seq_len, seed=1, dtype=th.float64)
    with th.no_grad():
        c_fit, c = run_with_cache(model, x_fit), run_with_cache(model, x)
    true = task_2nd_argmax(x)
    pred = c.logits.argmax(-1)
    acc = (pred == true).double().mean().item()
    S["accuracy"] = acc
    W_out = model.linear.weight.detach()

    # 1. surrogate rebuild
    S["surrogate"] = {}
    for level in LEVELS:
        h_hat = surrogate(model, x_fit, c_fit.h[:, -1], x, level)
        p = (h_hat @ W_out.T).argmax(-1)
        h = c.h[:, -1]
        S["surrogate"][level] = {
            "held_out_r2_of_final_state": (1 - ((h_hat - h) ** 2).sum() / ((h - h.mean(0)) ** 2).sum()).item(),
            "agreement_with_model": (p == pred).double().mean().item(),
            "accuracy": (p == true).double().mean().item(),
        }

    # 2. resample check
    x1, x2 = resampled(x, gen)
    assert (task_2nd_argmax(x1) == true).all() and (task_2nd_argmax(x2) == true).all()
    ok = pred == true
    S["resample"] = {"ceiling_if_independent_errors": acc ** 2}
    with th.no_grad():
        for key, xr in [("L1_irrelevant_inputs_below_2nd", x1), ("L2_keep_only_top2_positions", x2)]:
            pr = model(xr).argmax(-1)
            S["resample"][key] = {"unchanged": (pr == pred).double().mean().item(),
                                  "unchanged_where_model_right": (pr[ok] == pred[ok]).double().mean().item(),
                                  "accuracy_after": (pr == true).double().mean().item()}

    # 3. errors
    s = x.sort(-1, descending=True)
    order = s.indices
    g12, g23 = s.values[:, 0] - s.values[:, 1], s.values[:, 1] - s.values[:, 2]
    wrong = ~ok
    p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
    T = model.seq_len
    after = (p3 > p2) & (p3 > p1)
    S["errors"] = {
        "n_errors": int(wrong.sum()),
        "median_gap_max_2nd": {"all": g12.median().item(), "errors": g12[wrong].median().item()},
        "median_gap_2nd_3rd": {"all": g23.median().item(), "errors": g23[wrong].median().item()},
        "gap_2nd_3rd_5th_percentile_all": g23.quantile(0.05).item(),
        "frac_predicting_3rd": ((pred == p3) & wrong).sum().item() / wrong.sum().item(),
        "frac_predicting_3rd_arriving_after_max_and_2nd": ((pred == p3) & wrong & after).sum().item() / wrong.sum().item(),
        "frac_predicting_last_position_not_3rd": ((pred == T - 1) & (p3 != T - 1) & wrong).sum().item() / wrong.sum().item(),
        "errors_by_true_position": th.bincount(true[wrong], minlength=T).tolist(),
        "errors_by_predicted_position": th.bincount(pred[wrong], minlength=T).tolist(),
    }
    v2 = s.values[:, 1]
    # The model compares against max(0, ...), so it can't rank values below 0:
    # expect many errors when the 2nd largest is negative (P = 11/1024 for T = 10)
    neg2 = v2 < 0
    S["errors"]["second_largest_negative"] = {
        "prob": neg2.double().mean().item(),
        "prob_theory": (1 + T) / 2**T,
        "error_rate_when_negative": wrong[neg2].double().mean().item(),
        "error_rate_when_positive": wrong[~neg2].double().mean().item(),
        "share_of_errors": (wrong & neg2).sum().item() / wrong.sum().item(),
    }
    rel23 = (g23 / v2.abs().clamp(min=1e-12))
    for cut in [0.01, 0.03]:
        near = (~neg2) & (rel23 < cut)
        S["errors"][f"positive_2nd_near_tie_rel_gap_below_{cut}"] = {
            "prob": near.double().mean().item(),
            "error_rate": wrong[near].double().mean().item(),
            "share_of_errors": (wrong & near).sum().item() / wrong.sum().item(),
        }
    rest = (~neg2) & (rel23 >= 0.03)
    S["errors"]["positive_2nd_rel_gap_at_least_0.03"] = {
        "prob": rest.double().mean().item(), "error_rate": wrong[rest].double().mean().item(),
        "share_of_errors": (wrong & rest).sum().item() / wrong.sum().item()}
    edges = np.logspace(-4, 0.5, 28)
    b = np.digitize(rel23.numpy(), edges)
    rates, centers = [], []
    for i in range(1, len(edges)):
        sel = b == i
        if sel.sum() >= 200:
            rates.append(wrong.numpy()[sel].mean())
            centers.append(np.sqrt(edges[i - 1] * edges[i]))
    S["errors"]["error_rate_by_relative_gap_2nd_3rd"] = {"centers": centers, "rates": rates}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(centers, rates, marker="o", markersize=3, color=LINE_COLOR)
    ax.set_xscale("log")
    ax.set_xlabel("relative gap (2nd − 3rd) / |2nd|")
    ax.set_ylabel("error rate")
    ax.set_title("Errors are near-ties between the 2nd and 3rd largest")
    ax.grid(True, axis="y")
    savefig(fig, out / "errors.png")

    # 4. one-neuron error model (detector curve from the fit sample only)
    edges_t, f = detector_curve(c_fit, args.top2_neuron)
    case = after & (v2 > 0)
    rel_case = ((s.values[:, 1] - s.values[:, 2]) / v2)[case]
    predicted = f[th.bucketize(rel_case, edges_t, right=True) - 1].sum().item()
    S["one_neuron_error_model"] = {
        "detector_p_on_by_relative_shortfall": {"edges": SHORTFALL_EDGES, "p_on": f.tolist()},
        "predicted_errors_3rd_after_max_and_2nd": predicted,
        "measured_errors_3rd_after_max_and_2nd": int((wrong & case & (pred == p3)).sum()),
    }

    with open(out / "summary.json", "w") as fh:
        json.dump(S, fh, indent=2)
    print(json.dumps({k: v for k, v in S.items() if k != "errors"}, indent=2)[:3000])
    print(json.dumps({k: v for k, v in S["errors"].items() if k != "error_rate_by_relative_gap_2nd_3rd"}, indent=2))


if __name__ == "__main__":
    main()
