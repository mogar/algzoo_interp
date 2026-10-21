"""How neurons encode the lags of the max and 2nd max (where they are, not their values).

For every neuron and every lag k (steps since the running max, or 2nd max,
arrived):
- P(on | lag = k), pooled over steps t;
- t-invariance: the largest spread of P(on | lag = k, t) across steps t. A
  small spread means the pattern depends only on the lag, as expected of a
  clock restarted by each new max;
- for the max lag, a per-lag fit at the final step, h ≈ a + b*max + c*2nd + d*x_t,
  which shows whether the neuron carries the max or the 2nd max at that lag.

Writes results/<model>/lag_codes/{summary.json, p_on_lag_max.png,
p_on_lag_2nd.png, coefs_lag_max.png}.

Example:
    python scripts/lag_codes.py --hidden 32 --seq-len 10
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch as th

from algzoo_interp import randn_inputs, run_with_cache
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.plots import heatmap, savefig
from algzoo_interp.state import ideal_state, p_on_by_lag
from algzoo_interp.paths import RESULTS_DIR


def coefs_by_lag(cache, state, min_count, t=None):
    """Per-lag fit at the final step: h ≈ a + b*max + c*2nd + d*x_t. Returns b, c, d, R²: each [hidden, T]."""
    x = cache.x.cpu()
    T, H = x.shape[1], cache.h.shape[-1]
    t = T - 1 if t is None else t
    h = cache.h.cpu()[:, t].double()
    lag = state["lag_max"][:, t]
    b, c, d, r2 = (th.full((H, T), float("nan"), dtype=th.double) for _ in range(4))
    for k in range(T):
        sel = lag == k
        if int(sel.sum()) < min_count:
            continue
        X = th.stack([th.ones(int(sel.sum())), state["max"][sel, t], state["2nd"][sel, t], x[sel, t]], 1).double()
        Y = h[sel]
        coef = th.linalg.lstsq(X, Y).solution  # [4, H]
        res = ((X @ coef - Y) ** 2).sum(0)
        tot = ((Y - Y.mean(0)) ** 2).sum(0)
        # Neurons that are (almost) always off at this lag have no meaningful fit
        live = tot > 1e-6 * len(Y)
        nan = th.full_like(tot, float("nan"))
        b[:, k] = th.where(live, coef[1], nan)
        c[:, k] = th.where(live, coef[2], nan)
        d[:, k] = th.where(live, coef[3], nan)
        r2[:, k] = th.where(live, 1 - res / tot.clamp(min=1e-12), nan)
    return b, c, d, r2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**17)
    parser.add_argument("--min-count", type=int, default=200, help="min inputs per (t, lag) cell")
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.eval()
    out = RESULTS_DIR / name / "lag_codes"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Lag codes for {name} -> {out}")
    x = randn_inputs(args.n, model.seq_len)
    with th.no_grad():
        cache = run_with_cache(model, x)
    state = ideal_state(x)
    H = model.hidden_size

    summary = {"model": name, "n_inputs": args.n}
    for which in ["lag_max", "lag_2nd"]:
        pooled, spread = p_on_by_lag(cache, state[which], args.min_count)
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.25 * H + 2))
        heatmap(axes[0], pooled, title=f"P(on | {which} = k), pooled over steps",
                xlabel=f"{which} k", ylabel="neuron", vmin=0, vmax=1, annotate=H <= 32, fmt="{:.1f}")
        heatmap(axes[1], spread, title="Spread of P(on) across steps t (0 = depends on lag only)",
                xlabel=f"{which} k", ylabel="neuron", vmin=0, vmax=0.5)
        savefig(fig, out / f"p_on_{which}.png")
        finite = spread[~spread.isnan()]
        late = spread[:, 2:][~spread[:, 2:].isnan()]
        summary[which] = {
            "p_on": [[None if v != v else round(v, 3) for v in row] for row in pooled.tolist()],
            "spread": [[None if v != v else round(v, 3) for v in row] for row in spread.tolist()],
            "spread_median": round(finite.median().item(), 4),
            "spread_90pct": round(finite.quantile(0.9).item(), 4),
            "spread_lags_2plus_median": round(late.median().item(), 4),
            "spread_lags_2plus_90pct": round(late.quantile(0.9).item(), 4),
        }
        print(f"{which}: spread of P(on) across steps: median {summary[which]['spread_median']}, "
              f"90th pct {summary[which]['spread_90pct']}; lags >= 2: median "
              f"{summary[which]['spread_lags_2plus_median']}, 90th pct {summary[which]['spread_lags_2plus_90pct']}")

    b, c, d, r2 = coefs_by_lag(cache, state, args.min_count)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.25 * H + 2))
    lim = float(th.cat([b[~b.isnan()], c[~c.isnan()]]).abs().quantile(0.98))
    heatmap(axes[0], b, title="Final step: coefficient on the max (b)", xlabel="lag of max k",
            ylabel="neuron", diverging=True, lim=lim, annotate=H <= 32, fmt="{:.1f}", colorbar=False)
    heatmap(axes[1], c, title="Final step: coefficient on the 2nd max (c)", xlabel="lag of max k",
            diverging=True, lim=lim, annotate=H <= 32, fmt="{:.1f}")
    fig.suptitle("Per-lag fit h ≈ a + b·max + c·2nd + d·x_t (blank: neuron off at that lag)")
    savefig(fig, out / "coefs_lag_max.png")
    as_list = lambda m: [[None if v != v else round(v, 3) for v in row] for row in m.tolist()]  # noqa: E731
    summary["final_step_fit_by_lag_max"] = {"b_max": as_list(b), "c_2nd": as_list(c), "d_x_t": as_list(d), "r2": as_list(r2)}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
