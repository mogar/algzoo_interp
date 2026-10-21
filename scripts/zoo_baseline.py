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
from algzoo_interp.plots import heatmap, savefig

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

    # Largest hidden size in the top row
    hidden_sizes = sorted({r[0] for r in rows}, reverse=True)
    seq_lens = sorted({r[1] for r in rows})
    grid = np.full((len(hidden_sizes), len(seq_lens)), np.nan)
    for hidden_size, seq_len, _, acc in rows:
        grid[hidden_sizes.index(hidden_size), seq_lens.index(seq_len)] = acc

    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap(ax, grid, title=f"2nd argmax zoo accuracy (n={args.n:,} randn inputs)",
            xlabel="seq_len", ylabel="hidden_size", xticks=seq_lens, yticks=hidden_sizes,
            vmin=0, vmax=1, annotate=True, fmt="{:.1%}")
    savefig(fig, OUT_DIR / "accuracy.png")


if __name__ == "__main__":
    main()
