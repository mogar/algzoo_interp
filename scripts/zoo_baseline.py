"""Measure the accuracy of every 2nd argmax zoo model and save a baseline.

Writes results/zoo_baseline/{accuracy.csv,accuracy.png}.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from algzoo_interp import ZOO_2ND_ARGMAX_GRID, accuracy, load_2nd_argmax, randn_inputs

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "zoo_baseline"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2**20, help="inputs per model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for hidden_size, seq_len in ZOO_2ND_ARGMAX_GRID:
        model = load_2nd_argmax(hidden_size, seq_len).to(device)
        x = randn_inputs(args.n, seq_len, seed=args.seed, device=device)
        acc = accuracy(model, x)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append((hidden_size, seq_len, n_params, acc))
        print(f"hidden={hidden_size:2d} seq_len={seq_len:2d} params={n_params:5d} acc={acc:.4f}")

    with open(OUT_DIR / "accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_size", "seq_len", "n_params", "accuracy"])
        writer.writerows(rows)

    hidden_sizes = sorted({r[0] for r in rows})
    seq_lens = sorted({r[1] for r in rows})
    grid = np.full((len(hidden_sizes), len(seq_lens)), np.nan)
    for hidden_size, seq_len, _, acc in rows:
        grid[hidden_sizes.index(hidden_size), seq_lens.index(seq_len)] = acc

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=1)
    for i in range(len(hidden_sizes)):
        for j in range(len(seq_lens)):
            ax.text(j, i, f"{grid[i, j]:.1%}", ha="center", va="center",
                    color="white" if grid[i, j] < 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(seq_lens)), seq_lens)
    ax.set_yticks(range(len(hidden_sizes)), hidden_sizes)
    ax.set_xlabel("seq_len")
    ax.set_ylabel("hidden_size")
    ax.set_title(f"2nd argmax zoo accuracy (n={args.n:,} randn inputs)")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy.png", dpi=150)
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
