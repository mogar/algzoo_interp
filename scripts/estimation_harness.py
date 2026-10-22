"""Score error-rate estimators by mean squared error versus compute (docs/PLAN.md steps 9-11).

Builds a family of noisy copies of a model: the model itself, plus copies
with every weight w replaced by w + sigma * |w| * eps. For each copy it computes
a ground-truth error rate, sampled on --n-truth inputs and cached. For
seq_len-3 models the ground truth is exact, from cone enumeration. It then
scores the sampling baseline (E0) at a range of budgets, against the theory
MSE = mean over the family of p(1 - p) / N.

The ground truth has sampling noise of its own. Its known variance is
subtracted from every measured MSE (algzoo_interp.estimation.score), and the
remaining floor is reported.

It also scores E1 (step 10): exact cell probabilities times the error
sampled within each cell. The cells split inputs by sign (max < 0; only the
2nd < 0) and by the relative gaps max-2nd and 2nd-3rd
(algzoo_interp.order_stats). Inputs are split across cells by a Neyman
allocation from a pilot run on the unperturbed model. The allocation is a
fixed design choice and isn't counted as compute: any allocation keeps E1
unbiased. The cell table for the unperturbed model goes to cells.md.

It also scores E2 (step 11): the same cells, where each cell's error is
measured either by 0/1 samples or by exact integration along a tie line
through each sample (algzoo_interp.lines). The pilot picks whichever has the
smaller σ √cost (a line method must beat 0/1 samples by 20%, because line
values are heavy-tailed), and inputs are allocated by Neyman allocation with
costs.
As a diagnostic, it integrates every line exactly over the relative 2nd-3rd
near-tie window (γ = 0.03), to show how the tie error varies from line to
line (`e2_tie_window` in summary.json).

For seq_len-3 models it also checks the harness against exact cone
enumeration. The sampled ground truth should be within its standard error of
the exact value (mean z² ≈ 1). The exact method's compute is counted in
multiply-adds, like everything else.

Writes results/<model>/estimation/{ground_truth.json, summary.json, cells.md, mse_vs_compute.png}.

Example:
    python scripts/estimation_harness.py --hidden 32 --seq-len 10
    python scripts/estimation_harness.py --hidden 4 --seq-len 3 --n-truth 4194304
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch as th

from algzoo_interp import randn_inputs
from algzoo_interp.estimation import (
    CellLines,
    Stratified,
    design_cell_lines,
    errors_on,
    forward_macs,
    neyman_weights,
    noisy_family,
    score,
    true_error,
)
from algzoo_interp.lines import line_error_mass, tie_lines
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.order_stats import cell_prob, gap_cells
from algzoo_interp.paths import RESULTS_DIR
from algzoo_interp.plots import LINE_COLOR, savefig

TRUTH_SEED = 1_000_000
POOL_SEED = 2_000_000
E1_SEED = 3_000_000
E2_SEED = 4_000_000
G12_EDGES = [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]
G23_EDGES = [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, float("inf")]


def ground_truth(family, n_truth, path, label):
    cache = json.loads(path.read_text()) if path.exists() else {}
    for j, m in enumerate(family):
        key = f"{m.key}/n={n_truth}"
        if key not in cache:
            p, var = true_error(m.model, n_truth, seed=TRUTH_SEED + 1000 * j)
            cache[key] = {"error": p, "var": var}
            path.write_text(json.dumps(cache, indent=1))
            print(f"  ground truth {label} {m.key}: error {p:.5f} ± {var ** 0.5:.1e}")
    return [cache[f"{m.key}/n={n_truth}"] for m in family]


def exact_errors(family):
    from algzoo_interp import cones
    out = []
    for m in family:
        cones.op_count["macs"] = 0
        acc = cones.exact_accuracy(cones.enumerate_cones(m.model))
        out.append({"error": 1 - acc, "macs": cones.op_count["macs"]})
    return out


def score_estimator(label, estimator, budgets, family, truths, truth_vars, repeats, seed, macs1):
    """MSE over the family (repeats per model) at each budget, against sampling at the same compute."""
    rows = []
    for N in budgets:
        est, tru, var, macs = [], [], [], []
        for j, (m, t, v) in enumerate(zip(family, truths, truth_vars)):
            for r in range(repeats):
                e, c = estimator(m.model, N, seed=seed + 1000 * j + r)
                est.append(e)
                tru.append(t)
                var.append(v)
                macs.append(c)
        mse, floor = score(est, tru, var)
        mean_macs = sum(macs) / len(macs)
        theory = sum(t * (1 - t) for t in truths) / len(truths) / (mean_macs / macs1)
        rows.append({"budget": N, "macs": mean_macs, "mse": mse, "sampling_mse_same_compute": theory,
                     "gain": theory / mse if mse > 0 else None})
        print(f"  {label} N={N:>8}: MSE {mse:.3e}  sampling at same compute {theory:.3e}  gain {theory / mse:.2f}x")
    return rows


def tie_window_diagnostic(model, n, gamma=0.03, seed=5_000_000):
    """Exact tie error per line over the window |s| < γ |mid| / √2, on n random inputs."""
    x = randn_inputs(n, model.seq_len, seed=seed, dtype=th.float64)
    y, u, s, w, mid, others, j, k = tie_lines(x, (1, 2))
    eps = th.minimum(gamma * mid.abs() / 2**0.5, w)
    mass, passes = line_error_mass(model, y, u, -eps, eps, j, k)
    f = mass / (th.special.ndtr(w) - th.special.ndtr(-w))
    inside = errors_on(model, x) * (s.abs() < eps)
    q = f.quantile(th.tensor([0.5, 0.9, 0.99], dtype=th.float64)).tolist()
    return {"gamma": gamma, "n": n, "mean": f.mean().item(), "sd": f.std().item(),
            "sd_over_mean": (f.std() / f.mean()).item(), "quantiles_50_90_99": q,
            "bernoulli_sd": inside.std().item(), "cost_forward_passes": 2 * passes.double().mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.001, 0.003])
    parser.add_argument("--n-per", type=int, default=8, help="noisy copies per sigma")
    parser.add_argument("--n-truth", type=int, default=2**25)
    parser.add_argument("--pool", type=int, default=2**22, help="inputs per model for the sampling baseline")
    parser.add_argument("--budgets", nargs="+", type=int, default=[2**k for k in range(8, 21, 2)])
    parser.add_argument("--repeats", type=int, default=8, help="E1 estimates per model and budget")
    parser.add_argument("--pilot", type=int, default=2**14, help="E1 pilot inputs per cell")
    parser.add_argument("--e2-pilot", type=int, default=2**13, help="E2 pilot inputs per cell")
    parser.add_argument("--e2-max-budget", type=int, default=2**18, help="largest E2 budget (it is slow to run)")
    args = parser.parse_args()

    model, name = load_from_args(args)
    out = RESULTS_DIR / name / "estimation"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Estimation harness for {name} -> {out}")
    family = noisy_family(model, args.sigmas, args.n_per)
    macs1 = forward_macs(model)
    S = {"model": name, "sigmas": args.sigmas, "n_per": args.n_per, "forward_macs": macs1,
         "members": [m.key for m in family]}

    exact = exact_errors(family) if model.seq_len == 3 else None
    if exact is not None:
        # exact ground truth; the sampled one is only checked against it
        sampled = ground_truth(family, args.n_truth, out / "ground_truth.json", name)
        z2 = [(s["error"] - e["error"]) ** 2 / s["var"] for s, e in zip(sampled, exact) if s["var"] > 0]
        S["exact"] = {"errors": [e["error"] for e in exact], "macs": [e["macs"] for e in exact],
                      "sampled_truth_mean_z2": sum(z2) / len(z2)}
        truths = [e["error"] for e in exact]
        truth_vars = [0.0] * len(family)
        print(f"  exact vs sampled ground truth: mean z² = {S['exact']['sampled_truth_mean_z2']:.2f} (expect ≈ 1)")
    else:
        gt = ground_truth(family, args.n_truth, out / "ground_truth.json", name)
        truths, truth_vars = [g["error"] for g in gt], [g["var"] for g in gt]
    S["truth"] = {"errors": truths, "vars": truth_vars, "n_truth": None if exact is not None else args.n_truth}

    # E0: chunks of a per-model pool are independent sampling estimates at each budget
    pools = [errors_on(m.model, randn_inputs(args.pool, model.seq_len, seed=POOL_SEED + 1000 * j, dtype=th.float64))
             for j, m in enumerate(family)]
    rows = []
    for N in args.budgets:
        k = args.pool // N
        est, tru, var = [], [], []
        for pool, t, v in zip(pools, truths, truth_vars):
            chunks = pool[: k * N].view(k, N).mean(1).tolist()
            est += chunks
            tru += [t] * k
            var += [v] * k
        mse, floor = score(est, tru, var)
        theory = sum(t * (1 - t) for t in truths) / len(truths) / N
        rows.append({"budget": N, "macs": N * macs1, "mse": mse, "theory_mse": theory, "truth_var_floor": floor,
                     "estimates_per_model": k})
        print(f"  E0 N={N:>8}: MSE {mse:.3e}  theory {theory:.3e}  ratio {mse / theory:.2f}  (truth floor {floor:.1e})")
    S["sampling"] = rows

    # E1: stratified by cell, with exact cell probabilities
    T = model.seq_len
    cells = gap_cells(G12_EDGES, G23_EDGES)
    probs = [cell_prob(c, T) for c in cells]
    rates, weights = neyman_weights(family[0].model, cells, probs, T, args.pilot)
    e1 = Stratified(cells, probs, weights, T)
    total = sum(P * r for P, r in zip(probs, rates))
    lines = [f"# Error by cell: {name}", "",
             f"Generated by `python scripts/estimation_harness.py`: exact cell probabilities "
             f"(algzoo_interp.order_stats) and the unperturbed model's error on {args.pilot:,} inputs sampled in "
             f"each cell. g12 = (max - 2nd) / max and g23 = (2nd - 3rd) / 2nd. Cells under 0.5% of the error are "
             f"omitted. The Neyman weight is the share of E1's budget the cell gets.", "",
             f"Total error from the cells: {total:.5f}.", "",
             "| cell | probability | error rate | share of error | Neyman weight |", "|---|---|---|---|---|"]
    wsum = sum(weights)
    for c, P, r, w in sorted(zip(cells, probs, rates, weights), key=lambda z: -z[1] * z[2]):
        if P * r / total >= 0.005:
            lines.append(f"| {c.name} | {P:.3g} | {r:.4f} | {P * r / total:.3f} | {w / wsum:.3f} |")
    (out / "cells.md").write_text("\n".join(lines) + "\n")
    neyman_var = sum(weights) ** 2  # Σ_c P_c sqrt(p_c(1 - p_c)), squared
    plain_var = total * (1 - total)
    S["e1_design"] = {"cells": [c.name for c in cells], "probs": probs, "pilot_rates": rates,
                      "pilot_per_cell": args.pilot, "predicted_variance_ratio": plain_var / neyman_var}
    print(f"  E1: {len(cells)} cells, pilot error {total:.5f}; predicted gain over sampling "
          f"{plain_var / neyman_var:.2f}x (unperturbed model)")
    e1_rows = score_estimator("E1", e1, args.budgets, family, truths, truth_vars, args.repeats, E1_SEED, macs1)
    S["e1"] = e1_rows

    # E2: per-cell choice of 0/1 samples or exact tie-line integration
    methods, alloc, design_rows, predicted = design_cell_lines(family[0].model, cells, probs, T, args.e2_pilot)
    S["e2_design"] = {"pilot_per_cell": args.e2_pilot, "predicted_gain": predicted, "cells": design_rows,
                      "n_cells_by_method": {m: sum(r["method"] == m for r in design_rows)
                                            for m in sorted({r["method"] for r in design_rows})}}
    print(f"  E2: methods {S['e2_design']['n_cells_by_method']}; predicted gain over sampling {predicted:.2f}x")
    e2 = CellLines(cells, probs, methods, alloc, T)
    e2_rows = score_estimator("E2", e2, [N for N in args.budgets if N <= args.e2_max_budget], family, truths,
                              truth_vars, args.repeats, E2_SEED, macs1)
    S["e2"] = e2_rows
    if T > 3:
        S["e2_tie_window"] = tie_window_diagnostic(family[0].model, 2**16)
        d = S["e2_tie_window"]
        print(f"  tie window: per-line tie error mean {d['mean']:.5f}, sd/mean {d['sd_over_mean']:.1f}, "
              f"cost {d['cost_forward_passes']:.1f} forward passes")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.loglog([r["macs"] for r in rows], [r["theory_mse"] for r in rows], color=LINE_COLOR, label="sampling, theory p(1-p)/N")
    ax.loglog([r["macs"] for r in rows], [max(r["mse"], 1e-30) for r in rows], "o", color=LINE_COLOR,
              label="sampling, measured")
    ax.loglog([r["macs"] for r in e1_rows], [max(r["mse"], 1e-30) for r in e1_rows], "s", color="#009e73",
              label="E1: exact cell probabilities")
    ax.loglog([r["macs"] for r in e2_rows], [max(r["mse"], 1e-30) for r in e2_rows], "^", color="#cc79a7",
              label="E2: E1 + exact tie-line integration")
    if exact is not None:
        m_ex = sum(S["exact"]["macs"]) / len(family)
        ax.axvline(m_ex, color="#d55e00", ls="--", label=f"exact cones: MSE 0, {m_ex:.1e} MACs")
    ax.set_xlabel("compute (multiply-adds)")
    ax.set_ylabel("MSE of the error-rate estimate")
    ax.set_title(f"{name}: estimators over {len(family)} noisy models")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, out / "mse_vs_compute.png")
    (out / "summary.json").write_text(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
