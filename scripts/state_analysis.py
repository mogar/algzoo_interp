"""How the hidden state encodes the ideal online state, and how the readout uses it.

1. State sufficiency: for each neuron, R^2 of a fit using only the ideal
   online state (step, lag of max, lag of 2nd max), then adding the max and
   2nd max values, then adding the current input (see algzoo_interp/state.py).
2. Position codes: mean final hidden value of each important neuron for every
   (position of max, position of 2nd max) pair.
3. Readout map: share of each logit's variance contributed by each neuron.

Writes results/<model>/state/{summary.json, sufficiency.png, position_codes.png, readout.png}.

Example:
    python scripts/state_analysis.py --hidden 32 --seq-len 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from algzoo_interp import get_weights, randn_inputs, run_with_cache
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.plots import heatmap, savefig, to_numpy
from algzoo_interp.state import state_r2

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LEVELS = ["none", "values", "values+input"]
LEVEL_LABELS = {"none": "positions only", "values": "+ max, 2nd values", "values+input": "+ current input"}


def position_codes(cache, neurons):
    """Mean final h per (argmax position, 2nd-argmax position): [len(neurons), T, T]."""
    x = cache.x.cpu()
    T = x.shape[1]
    order = x.argsort(-1, descending=True)
    cell = order[:, 0] * T + order[:, 1]
    counts = th.bincount(cell, minlength=T * T).float()
    out = []
    for n in neurons:
        sums = th.bincount(cell, weights=cache.h[:, -1, n].cpu().double(), minlength=T * T)
        mean = (sums / counts).reshape(T, T)
        mean[counts.reshape(T, T) == 0] = float("nan")
        out.append(mean)
    return th.stack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--n", type=int, default=2**17)
    parser.add_argument("--n-codes", type=int, default=12, help="neurons to show position codes for")
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.eval()
    out = RESULTS_DIR / name / "state"
    out.mkdir(parents=True, exist_ok=True)
    print(f"State analysis for {name} -> {out}")
    x = randn_inputs(args.n, model.seq_len)
    with th.no_grad():
        cache = run_with_cache(model, x)
    T, H = model.seq_len, model.hidden_size
    steps = list(range(1, T))

    # 1. sufficiency
    r2 = {level: state_r2(cache, steps, level) for level in LEVELS}
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * H + 2), 3.5))
    width = 0.27
    colors = ["#9ec5f4", "#2a6fdb", "#0b3a82"]
    for k, (level, color) in enumerate(zip(LEVELS, colors)):
        ax.bar(np.arange(H) + (k - 1) * width, to_numpy(r2[level]), width * 0.9, color=color,
               label=LEVEL_LABELS[level])
    ax.set_xticks(range(H))
    ax.set_xlabel("neuron")
    ax.set_ylabel("R² (pooled over steps 1..T-1)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y")
    ax.set_title("How much of each neuron the ideal online state explains", pad=24)
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    savefig(fig, out / "sufficiency.png")

    # 3. readout map (computed before codes, to pick neurons that matter to the readout)
    w = get_weights(model)
    contrib = cache.h[:, -1, None, :] * w.W_out  # [batch, logits, neurons]
    var = contrib.var(0)
    share = (var / var.sum(1, keepdim=True)).detach()
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * H + 2), 0.35 * T + 1.5))
    heatmap(ax, share, title="Share of each logit's variance from each neuron (final step)",
            xlabel="neuron", ylabel="logit (position)", vmin=0, vmax=float(share.max()))
    savefig(fig, out / "readout.png")

    # 2. position codes for the neurons with the largest total readout share
    neurons = share.sum(0).argsort(descending=True)[: args.n_codes].tolist()
    codes = position_codes(cache, neurons)
    cols = 4
    rows = (len(neurons) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.0 * rows))
    for ax, n, code in zip(axes.flat, neurons, codes):
        heatmap(ax, code, title=f"neuron {n}", xlabel="2nd-argmax position",
                ylabel="argmax position", colorbar=True)
    for ax in list(axes.flat)[len(neurons):]:
        ax.axis("off")
    fig.suptitle("Mean final hidden value by position of the top two inputs")
    savefig(fig, out / "position_codes.png")

    # 4. position membership: P(on at final step | position p holds the max or 2nd max) vs not
    order = x.argsort(-1, descending=True)
    on_final = cache.mask[:, -1].cpu().float()
    p_in, p_out = th.zeros(H, T), th.zeros(H, T)
    for p in range(T):
        in_top2 = (order[:, 0] == p) | (order[:, 1] == p)
        p_in[:, p] = on_final[in_top2].mean(0)
        p_out[:, p] = on_final[~in_top2].mean(0)

    summary = {
        "p_on_final_given_pos_in_top2": [[round(v, 3) for v in row] for row in p_in.tolist()],
        "p_on_final_given_pos_not_in_top2": [[round(v, 3) for v in row] for row in p_out.tolist()],
        "model": name,
        "n_inputs": args.n,
        "state_r2": {level: [round(v, 4) for v in r2[level].tolist()] for level in LEVELS},
        "state_r2_median": {level: round(r2[level].median().item(), 4) for level in LEVELS},
        "readout_share": [[round(v, 4) for v in row] for row in share.tolist()],
        "position_code_neurons": neurons,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("median R²:", summary["state_r2_median"])
    for n in range(H):
        print(f"  neuron {n:2d}: " + "  ".join(f"{LEVEL_LABELS[lv]}={r2[lv][n]:.3f}" for lv in LEVELS))


if __name__ == "__main__":
    main()
