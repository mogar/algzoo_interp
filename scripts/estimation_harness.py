"""Score error-rate estimators by mean squared error versus compute (docs/PLAN.md step 9).

Builds a family of noisy copies of a model: the model itself, plus copies
with every weight w replaced by w + sigma * |w| * eps. For each copy it computes
a ground-truth error rate, sampled on --n-truth inputs and cached. For
seq_len-3 models the ground truth is exact, from cone enumeration. It then
scores the sampling baseline (E0) at a range of budgets, against the theory
MSE = mean over the family of p(1 - p) / N.

The ground truth has sampling noise of its own. Its known variance is
subtracted from every measured MSE (algzoo_interp.estimation.score), and the
remaining floor is reported.

For seq_len-3 models it also checks the harness against exact cone
enumeration. The sampled ground truth should be within its standard error of
the exact value (mean z² ≈ 1). The exact method's compute is counted in
multiply-adds, like everything else.

Writes results/<model>/estimation/{ground_truth.json, summary.json, mse_vs_compute.png}.

Example:
    python scripts/estimation_harness.py --hidden 32 --seq-len 10
    python scripts/estimation_harness.py --hidden 4 --seq-len 3 --n-truth 4194304
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch as th

from algzoo_interp import randn_inputs
from algzoo_interp.estimation import errors_on, forward_macs, noisy_family, score, true_error
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.paths import RESULTS_DIR
from algzoo_interp.plots import LINE_COLOR, savefig

TRUTH_SEED = 1_000_000
POOL_SEED = 2_000_000


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.001, 0.003])
    parser.add_argument("--n-per", type=int, default=8, help="noisy copies per sigma")
    parser.add_argument("--n-truth", type=int, default=2**25)
    parser.add_argument("--pool", type=int, default=2**22, help="inputs per model for the sampling baseline")
    parser.add_argument("--budgets", nargs="+", type=int, default=[2**k for k in range(8, 21, 2)])
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

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.loglog([r["macs"] for r in rows], [r["theory_mse"] for r in rows], color=LINE_COLOR, label="sampling, theory p(1-p)/N")
    ax.loglog([r["macs"] for r in rows], [max(r["mse"], 1e-30) for r in rows], "o", color=LINE_COLOR,
              label="sampling, measured")
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
