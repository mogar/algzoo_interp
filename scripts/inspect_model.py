"""First-look inspection of one 2nd argmax model using the Phase 1 tools.

Writes figures and summary.json to results/<model>/inspect/:
- weights.png          W_ih, W_hh, W_out
- acc_by_positions.png accuracy by (argmax position, 2nd-argmax position)
- acc_by_gap.png       accuracy vs. gap between the top two values
- activity.png         fraction of inputs where each neuron is active, per step
- patterns.png         how many activation patterns cover the input distribution
- ablations.png        accuracy after ablating each neuron at all steps
- traces.png           hidden states over time for a few example inputs

Examples:
    python scripts/inspect_model.py --hidden 4 --seq-len 3
    python scripts/inspect_model.py --example
    python scripts/inspect_model.py --handcrafted 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from alg_zoo import handcrafted_2nd_argmax
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import (
    ablate,
    accuracy,
    accuracy_by_top2_gap,
    accuracy_by_top2_positions,
    activity,
    count_patterns,
    get_weights,
    load_2nd_argmax,
    load_example_2nd_argmax,
    randn_inputs,
    run_with_cache,
)
from algzoo_interp.plots import LINE_COLOR, heatmap, savefig, to_numpy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load(args):
    if args.example:
        return load_example_2nd_argmax(), "example_16x10"
    if args.handcrafted:
        return handcrafted_2nd_argmax(args.handcrafted), f"handcrafted_{args.handcrafted}"
    if args.hidden is None or args.seq_len is None:
        raise SystemExit("pass --hidden and --seq-len, or --example, or --handcrafted")
    name = f"2nd_argmax_{args.hidden}x{args.seq_len}"
    if args.seed is not None:
        name += f"_seed{args.seed}"
    if args.n_seqs is not None:
        name += f"_nseqs{args.n_seqs}"
    return load_2nd_argmax(args.hidden, args.seq_len, n_seqs=args.n_seqs, seed=args.seed), name


def plot_weights(model, out):
    w = get_weights(model)
    hidden = w.W_ih.shape[0]
    # One shared colour scale so magnitudes are comparable across panels, clipped
    # at the 99th percentile of |w| so a few large weights don't wash out the rest
    all_w = th.cat([w.W_ih.flatten(), w.W_hh.flatten(), w.W_out.flatten()]).abs()
    lim = all_w.quantile(0.99).item()
    fig, axes = plt.subplots(1, 3, figsize=(4 + hidden * 0.35, 3 + hidden * 0.12),
                             gridspec_kw={"width_ratios": [1, hidden, w.W_out.shape[0]]})
    fig.suptitle(f"Weights (colour clipped at ±{lim:.1f}; max |w| = {all_w.max():.1f})")
    heatmap(axes[0], w.W_ih[:, None], title="W_ih", ylabel="neuron", xticks=[""],
            diverging=True, lim=lim, colorbar=False)
    heatmap(axes[1], w.W_hh, title="W_hh (row = to, col = from)", xlabel="from neuron",
            diverging=True, lim=lim, colorbar=False)
    heatmap(axes[2], w.W_out.T, title="W_out^T", xlabel="output position",
            diverging=True, lim=lim)
    savefig(fig, out / "weights.png")


def plot_positions(model, x, out):
    acc, counts = accuracy_by_top2_positions(model, x)
    seq_len = x.shape[1]
    fig, ax = plt.subplots(figsize=(1.5 + 0.55 * seq_len, 1 + 0.5 * seq_len))
    heatmap(ax, acc, title="Accuracy by position of top two",
            xlabel="2nd-argmax position", ylabel="argmax position",
            vmin=0, vmax=1, annotate=seq_len <= 10, fmt="{:.2f}")
    savefig(fig, out / "acc_by_positions.png")
    return to_numpy(acc)


def plot_gap(model, x, out):
    edges = th.cat([th.zeros(1), th.logspace(-3, 0.5, 30)])
    acc, counts = accuracy_by_top2_gap(model, x, edges)
    keep = to_numpy(counts) >= 50
    centers = to_numpy((edges[:-1] + edges[1:]) / 2)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(centers[keep], to_numpy(acc)[keep], color=LINE_COLOR, marker="o", markersize=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y")
    ax.set_xlabel("top-two gap (largest − 2nd largest)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs. top-two gap (bins with ≥50 inputs)")
    savefig(fig, out / "acc_by_gap.png")


def plot_activity(cache, out):
    act = activity(cache)
    seq_len, hidden = act.shape
    fig, ax = plt.subplots(figsize=(1.5 + 0.3 * hidden, 1 + 0.35 * seq_len))
    heatmap(ax, act, title="Fraction of inputs with neuron active",
            xlabel="neuron", ylabel="step t", vmin=0, vmax=1,
            annotate=hidden <= 16, fmt="{:.1f}")
    savefig(fig, out / "activity.png")
    act = to_numpy(act)
    return {
        "dead": [int(i) for i in np.where((act == 0).all(0))[0]],
        "always_on": [int(i) for i in np.where((act == 1).all(0))[0]],
    }


def plot_patterns(cache, out):
    seq_len = cache.x.shape[1]
    full = count_patterns(cache)
    last = count_patterns(cache, steps=[seq_len - 1])
    fig, ax = plt.subplots(figsize=(5, 3))
    for pc, label, style in [(full, "all steps", "-"), (last, "last step only", "--")]:
        cov = to_numpy(pc.coverage)
        ax.plot(np.arange(1, len(cov) + 1), cov, style, color=LINE_COLOR, label=label)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y")
    ax.set_xlabel("number of most common patterns")
    ax.set_ylabel("fraction of inputs covered")
    ax.set_title("Activation-pattern coverage")
    ax.legend(frameon=False)
    savefig(fig, out / "patterns.png")
    cov = to_numpy(full.coverage)
    return {
        "n_patterns": len(full.counts),
        "n_patterns_last_step": len(last.counts),
        "patterns_for_90pct": int(np.searchsorted(cov, 0.9) + 1),
        "patterns_for_99pct": int(np.searchsorted(cov, 0.99) + 1),
    }


def plot_ablations(model, x, base_acc, out):
    hidden = model.hidden_size
    target = task_2nd_argmax(x)
    results = {}
    for mode in ["zero", "mean", "resample"]:
        results[mode] = [
            (ablate(model, x, neurons=[n], mode=mode).logits.argmax(-1) == target).float().mean().item()
            for n in range(hidden)
        ]
    fig, ax = plt.subplots(figsize=(max(5, 2 + 0.3 * hidden), 3.5))
    width = 0.27
    colors = {"zero": "#9ec5f4", "mean": LINE_COLOR, "resample": "#0b3a82"}
    for k, mode in enumerate(results):
        ax.bar(np.arange(hidden) + (k - 1) * width, base_acc - np.array(results[mode]),
               width=width * 0.9, color=colors[mode], label=f"{mode} ablation")
    ax.set_xticks(range(hidden))
    ax.set_xlabel("neuron (ablated at all steps)")
    ax.set_ylabel("accuracy drop")
    ax.set_title(f"Single-neuron ablations (baseline acc {base_acc:.3f})")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.08), fontsize=8)
    savefig(fig, out / "ablations.png")
    return results


def plot_traces(model, out, n_examples=4):
    x = randn_inputs(n_examples, model.seq_len, seed=123)
    cache = run_with_cache(model, x)
    fig, axes = plt.subplots(1, n_examples, figsize=(3.2 * n_examples, 3.2))
    for i, ax in enumerate(axes):
        target = x[i].argsort()[-2].item()
        pred = cache.logits[i].argmax().item()
        xs = ", ".join(f"{v:.2f}" for v in x[i].tolist())
        heatmap(ax, cache.h[i].T, xlabel="step t", ylabel="neuron" if i == 0 else "",
                colorbar=i == n_examples - 1,
                title=f"x = [{xs}]\ntarget {target}, predicted {pred}")
        ax.title.set_fontsize(7)
    savefig(fig, out / "traces.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--seed", type=int, help="training seed (default: best of 5)")
    parser.add_argument("--n-seqs", type=int, help="checkpoint (default: fully trained)")
    parser.add_argument("--example", action="store_true", help="blog post 16x10 model")
    parser.add_argument("--handcrafted", type=int, choices=[2, 3, 10])
    parser.add_argument("--n", type=int, default=2**18, help="randn inputs to evaluate on")
    parser.add_argument("--n-ablation", type=int, default=2**15, help="inputs per ablation")
    args = parser.parse_args()

    model, name = load(args)
    model = model.eval()
    out = RESULTS_DIR / name / "inspect"
    print(f"Inspecting {name} -> {out}")

    with th.no_grad():
        x = randn_inputs(args.n, model.seq_len)
        cache = run_with_cache(model, x)
        base_acc = accuracy(model, x)
        plot_weights(model, out)
        acc_pos = plot_positions(model, x, out)
        plot_gap(model, x, out)
        neuron_activity = plot_activity(cache, out)
        patterns = plot_patterns(cache, out)
        x_abl = x[: args.n_ablation]
        ablations = plot_ablations(model, x_abl, accuracy(model, x_abl), out)
        plot_traces(model, out)

    worst = np.unravel_index(np.nanargmin(acc_pos), acc_pos.shape)
    summary = {
        "model": name,
        "hidden_size": model.hidden_size,
        "seq_len": model.seq_len,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_inputs": args.n,
        "accuracy": base_acc,
        "worst_top2_positions": {"argmax_pos": int(worst[0]), "second_pos": int(worst[1]),
                                 "accuracy": float(acc_pos[worst])},
        **neuron_activity,
        **patterns,
        "single_neuron_ablation_accuracy": ablations,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "single_neuron_ablation_accuracy"}, indent=2))


if __name__ == "__main__":
    main()
