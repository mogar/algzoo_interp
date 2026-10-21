"""Predict the model's error on shifted inputs from its error mechanisms (H8).

Adding a constant c to every input leaves the true answer unchanged, but
changes how often the model's two blind spots occur: the 2nd largest being
negative (max(0, ...) comparisons can't rank values below 0) and near-ties
between the 2nd and 3rd largest relative to the 2nd's size (comparisons have
finite relative resolution).

The prediction uses only the error breakdown at c = 0: the error rates in
four input classes (max and 2nd negative; only the 2nd negative; 2nd
positive with a relative 2nd-3rd gap below 3%; everything else),
re-weighted by how often each class occurs at each shift, computed from the
inputs alone. It is compared with the
model's measured error at each shift.

Writes results/<model>/shift/{summary.json, shift.png}.

Example:
    python scripts/shift_test.py --hidden 32 --seq-len 10
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch as th
from alg_zoo.tasks import task_2nd_argmax

from algzoo_interp import randn_inputs
from algzoo_interp.loading import add_model_args, load_from_args
from algzoo_interp.paths import RESULTS_DIR
from algzoo_interp.plots import LINE_COLOR, savefig

CLASSES = ["max and 2nd negative", "2nd negative, max positive", "near-tie (rel gap < 3%)", "other"]


def input_class(x, rel_cut=0.03):
    """Index into CLASSES for each input."""
    s = x.sort(-1, descending=True).values
    v1, v2, v3 = s[:, 0], s[:, 1], s[:, 2]
    near = (v2 >= 0) & ((v2 - v3) / v2.clamp(min=1e-12) < rel_cut)
    return th.where(v1 < 0, 0, th.where(v2 < 0, 1, th.where(near, 2, 3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--shifts", nargs="+", type=float, default=[-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2])
    parser.add_argument("--n", type=int, default=2**19)
    args = parser.parse_args()

    model, name = load_from_args(args)
    model = model.double().eval()
    out = RESULTS_DIR / name / "shift"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Shift test for {name} -> {out}")
    x = randn_inputs(args.n, model.seq_len, seed=1, dtype=th.float64)

    def errors(xx):
        with th.no_grad():
            return (model(xx).argmax(-1) != task_2nd_argmax(xx)).double()

    # error rate per class at c = 0, measured on a separate sample
    x_cal = randn_inputs(args.n, model.seq_len, seed=0, dtype=th.float64)
    err_cal, cls_cal = errors(x_cal), input_class(x_cal)
    class_error = [err_cal[cls_cal == k].mean().item() for k in range(len(CLASSES))]
    class_count = [int((cls_cal == k).sum()) for k in range(len(CLASSES))]

    print("error rate by class at shift 0:", {k: (round(e, 4), n) for k, e, n in zip(CLASSES, class_error, class_count)})
    rows = []
    for c in args.shifts:
        xs = x + c
        cls = input_class(xs)
        freq = [(cls == k).double().mean().item() for k in range(len(CLASSES))]
        predicted = sum(f * e for f, e in zip(freq, class_error))
        measured = errors(xs).mean().item()
        rows.append({"shift": c, "class_freq": freq, "predicted_error": predicted, "measured_error": measured})
        print(f"shift {c:+5.2f}: class freq {[round(f, 4) for f in freq]}  predicted error {predicted:.4f}  measured {measured:.4f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    cs = [r["shift"] for r in rows]
    ax.plot(cs, [r["measured_error"] for r in rows], marker="o", color=LINE_COLOR, label="measured")
    ax.plot(cs, [r["predicted_error"] for r in rows], marker="s", ls="--", color="#d55e00",
            label="predicted from the shift-0 error breakdown")
    ax.set_yscale("log")
    ax.set_xlabel("shift c added to every input")
    ax.set_ylabel("error rate")
    ax.set_title("Error under shifted inputs: prediction vs measurement")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, out / "shift.png")
    with open(out / "summary.json", "w") as f:
        json.dump({"model": name, "n_inputs": args.n, "classes": CLASSES, "class_error_at_shift_0": class_error,
                   "class_count_at_shift_0": class_count,
                   "by_shift": rows}, f, indent=2)


if __name__ == "__main__":
    main()
