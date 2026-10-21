"""Causal tests of the event-detector, reset and veto hypotheses (H1, H2, H5, H6).

H1/H2 (single-step patching). Build input pairs that differ only at step t,
where the src input's x_t is changed so that an event fires or stops firing:
  add_max      clean x_t is not a new positive max; src x_t is
  remove_max   clean x_t is a new positive max; src x_t is not
  add_top2     clean x_t is not a new positive top-2 value; src x_t is (but not a new max)
  remove_top2  clean x_t is a new positive top-2 value (not a new max); src x_t isn't
Later inputs are identical, so the hidden state at step t fully determines
the final prediction: patching *all* neurons at step t reproduces src's
prediction exactly. Patching subsets shows which neurons carry the change.
We report the flip rate: P(patched prediction == src prediction), over pairs
where the clean and src predictions differ.

H5 (resets). For each source neuron, targets are neurons j with
W_hh[j, source] < -reset_threshold. We measure P(target is 0 at t+1) when the
source is on vs off at t, and the effect of removing those edges.

H6 (vetoes). Remove single readout weights (e.g. neuron 7 -> logit 9) and
measure accuracy overall and by argmax position.

Writes results/<model>/causal/{summary.json, flip_rates.png, resets.png, veto.png}.

Example:
    python scripts/causal_tests.py --hidden 32 --seq-len 10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import ablate_edges, get_weights, patch, randn_inputs, run_with_cache
from algzoo_interp.features import running_top2
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.plots import LINE_COLOR, savefig

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
KINDS = ["add_max", "remove_max", "add_top2", "remove_top2"]
SUBSET_COLORS = ["#0b3a82", "#2a6fdb", "#9ec5f4", "#d55e00", "#e69f00"]


def make_pairs(x, t, kind, seed=0):
    """Clean inputs (filtered) and src copies differing only at step t."""
    gen = th.Generator().manual_seed(seed + 1000 * t)
    top1, top2 = running_top2(x)
    max0_prev = top1[:, t - 1].clamp(min=0)
    top2_prev0 = top2[:, t - 1].clamp(min=0)
    xt = x[:, t]
    new_max = xt > max0_prev
    new_top2 = (xt > top2_prev0) & ~new_max
    noise = th.randn(len(x), generator=gen).abs() * 0.5 + 0.1
    if kind == "add_max":
        keep, src_val = ~new_max, max0_prev + noise
    elif kind == "remove_max":
        keep, src_val = new_max, max0_prev - noise
    elif kind == "add_top2":
        # need room between the top-2 and the max for a new top-2 value
        room = top1[:, t - 1] - top2_prev0
        keep = ~new_max & ~new_top2 & (room > 0.2)
        src_val = top2_prev0 + th.rand(len(x), generator=gen) * room
    elif kind == "remove_top2":
        keep, src_val = new_top2, top2_prev0 - noise
    else:
        raise ValueError(kind)
    clean = x[keep]
    src = clean.clone()
    src[:, t] = src_val[keep]
    return clean, src


def flip_rates(model, x, subsets, steps, seed):
    results = {kind: {name: [] for name in subsets} for kind in KINDS}
    informative = {kind: [] for kind in KINDS}
    for kind in KINDS:
        for t in steps:
            clean, src = make_pairs(x, t, kind, seed)
            with th.no_grad():
                pred_clean = run_with_cache(model, clean).logits.argmax(-1)
                pred_src = run_with_cache(model, src).logits.argmax(-1)
            differ = pred_clean != pred_src
            informative[kind].append({"t": t, "n_pairs": len(clean), "frac_differ": differ.float().mean().item()})
            for name, neurons in subsets.items():
                pred = patch(model, clean[differ], src[differ], neurons=neurons, steps=[t]).logits.argmax(-1)
                results[kind][name].append((pred == pred_src[differ]).float().mean().item())
    return results, informative


def reset_stats(model, x, sources, threshold):
    w = get_weights(model)
    target = task_2nd_argmax(x)
    with th.no_grad():
        cache = run_with_cache(model, x)
        base_acc = (cache.logits.argmax(-1) == target).float().mean().item()
    out = {}
    for s in sources:
        targets = [int(j) for j in th.where(w.W_hh[:, s] < -threshold)[0]]
        on = cache.mask[:, :-1, s]  # source on at t, for t = 0..T-2
        rows = []
        edited = ablate_edges(model, hh=[(j, s) for j in targets])
        with th.no_grad():
            cut = run_with_cache(edited, x)
        for j in targets:
            zero_next = cache.h[:, 1:, j] <= 1e-6  # target at t+1
            zero_next_cut = cut.h[:, 1:, j] <= 1e-6
            rows.append({
                "target": j,
                "weight": round(w.W_hh[j, s].item(), 2),
                "p_zero_given_on": zero_next[on].float().mean().item(),
                "p_zero_given_off": zero_next[~on].float().mean().item(),
                "p_zero_given_on_edges_cut": zero_next_cut[cut.mask[:, :-1, s]].float().mean().item(),
            })
        out[s] = {
            "targets": rows,
            "accuracy": base_acc,
            "accuracy_edges_cut": (cut.logits.argmax(-1) == target).float().mean().item(),
        }
    return out


def veto_stats(model, x, edges):
    target = task_2nd_argmax(x)
    argmax_pos = x.argmax(-1)
    variants = {"baseline": model}
    for e in edges:
        variants[f"cut {e[1]}->logit {e[0]}"] = ablate_edges(model, out=[e])
    if len(edges) > 1:
        variants["cut all"] = ablate_edges(model, out=edges)
    out = {}
    for name, m in variants.items():
        with th.no_grad():
            pred = run_with_cache(m, x).logits.argmax(-1)
        correct = pred == target
        by_pos = [correct[argmax_pos == p].float().mean().item() for p in range(x.shape[1])]
        wrong = ~correct
        # Rank of the last input within its sequence (0 = smallest, T-1 = largest),
        # among errors that wrongly predict the last position
        last_rank = (x < x[:, -1:]).sum(-1)
        wrong_last = wrong & (pred == x.shape[1] - 1)
        rank_hist = th.bincount(last_rank[wrong_last], minlength=x.shape[1]).tolist()
        out[name] = {
            "accuracy": correct.float().mean().item(),
            "accuracy_by_argmax_pos": by_pos,
            "frac_errors_with_argmax_at_last": (wrong & (argmax_pos == x.shape[1] - 1)).sum().item() / max(wrong.sum().item(), 1),
            "frac_errors_predicting_last": (wrong & (pred == x.shape[1] - 1)).sum().item() / max(wrong.sum().item(), 1),
            "rank_of_last_when_wrongly_predicted_hist": rank_hist,
        }
    return out


def plot_flips(results, steps, path):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(4 * len(KINDS), 3.6), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for (name, rates), color in zip(results[kind].items(), SUBSET_COLORS):
            ax.plot(steps, rates, marker="o", markersize=4, color=color, label=name)
        ax.set_title(kind)
        ax.set_xlabel("patched step t")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("flip rate (patched pred = src pred)")
    axes[-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    savefig(fig, path)


def plot_resets(resets, path):
    fig, axes = plt.subplots(1, len(resets), figsize=(5 * len(resets), 3.4), squeeze=False)
    for ax, (s, info) in zip(axes[0], resets.items()):
        rows = info["targets"]
        idx = np.arange(len(rows))
        width = 0.27
        for k, (key, label, color) in enumerate([
            ("p_zero_given_off", f"{s} off at t", "#9ec5f4"),
            ("p_zero_given_on", f"{s} on at t", LINE_COLOR),
            ("p_zero_given_on_edges_cut", f"{s} on, edges cut", "#d55e00"),
        ]):
            ax.bar(idx + (k - 1) * width, [r[key] for r in rows], width * 0.9, color=color, label=label)
        ax.set_xticks(idx, [f"{r['target']}\n({r['weight']})" for r in rows], fontsize=7)
        ax.set_xlabel("target neuron (W_hh weight from source)")
        ax.set_ylabel("P(target = 0 at t+1)")
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="y")
        ax.set_title(f"Resets from neuron {s}", pad=24)
        ax.legend(frameon=False, fontsize=8, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    savefig(fig, path)


def plot_veto(veto, seq_len, path):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = ["#0b3a82", "#2a6fdb", "#9ec5f4", "#d55e00"]
    for (name, info), color in zip(veto.items(), colors):
        # "cut all" coincides with single cuts over some positions; dash it so both stay visible
        style = {"ls": "--", "lw": 1.5, "zorder": 3} if name == "cut all" else {}
        ax.plot(range(seq_len), info["accuracy_by_argmax_pos"], marker="o", markersize=4,
                color=color, label=f"{name} ({info['accuracy']:.3f})", **style)
    ax.set_xlabel("position of the largest input")
    ax.set_ylabel("accuracy")
    ax.set_title("Readout edge ablations")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--max-neuron", type=int, default=7, help="new-max detector (H1)")
    parser.add_argument("--top2-neuron", type=int, default=16, help="new-top-2 detector (H2)")
    parser.add_argument("--veto", nargs="+", default=["7:9", "12:9"], help="readout edges neuron:logit (H6)")
    parser.add_argument("--reset-threshold", type=float, default=5.0)
    parser.add_argument("--n", type=int, default=2**16)
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.eval()
    out = RESULTS_DIR / name / "causal"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Causal tests for {name} -> {out}")
    x = randn_inputs(args.n, model.seq_len)
    m, t2 = args.max_neuron, args.top2_neuron
    others = [n for n in range(model.hidden_size) if n not in (m, t2)]
    subsets = {
        f"{{{m}}}": [m],
        f"{{{t2}}}": [t2],
        f"{{{m}, {t2}}}": [m, t2],
        f"all but {m}": [n for n in range(model.hidden_size) if n != m],
        f"all but {m}, {t2}": others,
    }
    steps = list(range(2, model.seq_len))

    flips, informative = flip_rates(model, x, subsets, steps, seed=0)
    resets = reset_stats(model, x, [m, t2], args.reset_threshold)
    edges = [(int(e.split(":")[1]), int(e.split(":")[0])) for e in args.veto]
    veto = veto_stats(model, x, edges)

    plot_flips(flips, steps, out / "flip_rates.png")
    plot_resets(resets, out / "resets.png")
    plot_veto(veto, model.seq_len, out / "veto.png")
    with open(out / "summary.json", "w") as f:
        json.dump({"model": name, "n_inputs": args.n, "steps": steps, "flip_rates": flips,
                   "pairs": informative, "resets": resets, "veto": veto}, f, indent=2)

    print("\nFlip rates (mean over steps):")
    for kind in KINDS:
        mean_differ = np.mean([i["frac_differ"] for i in informative[kind]])
        print(f"  {kind:12s} (clean/src predictions differ in {mean_differ:.0%} of pairs)")
        for sub, rates in flips[kind].items():
            print(f"    patch {sub:14s}: {np.mean(rates):.3f}   per step {[round(r, 2) for r in rates]}")
    print("\nResets:")
    for s, info in resets.items():
        print(f"  source {s}: accuracy {info['accuracy']:.4f} -> {info['accuracy_edges_cut']:.4f} with edges cut")
        for r in info["targets"]:
            print(f"    -> {r['target']:2d} (w={r['weight']:6.2f}): P(0|on)={r['p_zero_given_on']:.3f}"
                  f"  P(0|off)={r['p_zero_given_off']:.3f}  P(0|on, cut)={r['p_zero_given_on_edges_cut']:.3f}")
    print("\nVetoes:")
    for vname, info in veto.items():
        print(f"  {vname:20s} acc={info['accuracy']:.4f}  acc|argmax@last={info['accuracy_by_argmax_pos'][-1]:.4f}"
              f"  errors with argmax@last={info['frac_errors_with_argmax_at_last']:.2f}"
              f"  errors predicting last={info['frac_errors_predicting_last']:.2f}"
              f"\n  {'':20s} rank of last input when last is wrongly predicted (0=smallest): "
              f"{info['rank_of_last_when_wrongly_predicted_hist']}")


if __name__ == "__main__":
    main()
