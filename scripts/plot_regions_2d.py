"""Draw the activation-pattern regions of a seq_len=2 model in the input plane.

With seq_len 2 each input is a point (x_0, x_1), so the cones that the model is
linear on can be drawn directly. Writes results/geometry/regions_<hidden>x2.png
with two panels: the regions coloured by activation pattern, and the model's
prediction against the true answer.

Example:
    python scripts/plot_regions_2d.py --hidden 2
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.colors import ListedColormap

from algzoo_interp import count_patterns, load_2nd_argmax, run_with_cache
from algzoo_interp.paths import RESULTS_DIR
from algzoo_interp.plots import savefig

OUT_DIR = RESULTS_DIR / "geometry"

# Okabe-Ito: a categorical palette that stays distinguishable under colour blindness
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def pattern_label(pattern):
    return " | ".join(f"t{t}:" + "".join(str(int(b)) for b in row) for t, row in enumerate(pattern))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--extent", type=float, default=3.0)
    parser.add_argument("--resolution", type=int, default=600)  # even, so no grid point lies exactly on an axis
    args = parser.parse_args()

    model = load_2nd_argmax(args.hidden, 2).double()
    lin = np.linspace(-args.extent, args.extent, args.resolution)
    g0, g1 = np.meshgrid(lin, lin)
    x = th.tensor(np.stack([g0.ravel(), g1.ravel()], 1))
    with th.no_grad():
        cache = run_with_cache(model, x)
    pc = count_patterns(cache)
    if len(pc.counts) > len(CATEGORICAL):
        raise SystemExit(f"{len(pc.counts)} patterns; too many to colour distinctly")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    ax = axes[0]
    regions = pc.inverse.numpy().reshape(g0.shape)
    cmap = ListedColormap(CATEGORICAL[: len(pc.counts)])
    ax.imshow(regions, origin="lower", extent=[-args.extent, args.extent] * 2,
              cmap=cmap, vmin=-0.5, vmax=len(pc.counts) - 0.5, alpha=0.75, interpolation="nearest")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[k], alpha=0.75)
               for k in range(len(pc.counts))]
    ax.legend(handles, [pattern_label(p) for p in pc.patterns], title="pattern (on=1)",
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    ax.set_title(f"Activation-pattern regions, {args.hidden}x2 model")

    ax = axes[1]
    pred = cache.logits.argmax(-1).numpy().reshape(g0.shape)
    ax.imshow(pred, origin="lower", extent=[-args.extent, args.extent] * 2,
              cmap=ListedColormap(["#d6e6f7", "#f6d7c3"]), vmin=-0.5, vmax=1.5, interpolation="nearest")
    ax.plot([-args.extent, args.extent], [-args.extent, args.extent], color="black", lw=1.5,
            ls="--", label="true boundary x_0 = x_1")
    ax.text(-args.extent * 0.9, args.extent * 0.8, "model predicts 0", fontsize=9)
    ax.text(args.extent * 0.3, -args.extent * 0.5, "model predicts 1", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Model prediction (2nd argmax position)")

    for ax in axes:
        ax.axhline(0, color="#999999", lw=0.6)
        ax.axvline(0, color="#999999", lw=0.6)
        ax.set_xlabel("x_0")
        ax.set_ylabel("x_1")
        ax.set_aspect("equal")
    fig.subplots_adjust(wspace=0.6)
    savefig(fig, OUT_DIR / f"regions_{args.hidden}x2.png")


if __name__ == "__main__":
    main()
