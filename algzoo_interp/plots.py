"""Shared matplotlib helpers so every script's figures look the same.

Colour rules: magnitudes use one sequential hue (Blues); signed values use a
diverging map (RdBu_r) centred on 0 with a neutral midpoint.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
LINE_COLOR = "#2a6fdb"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "font.size": 9,
        "axes.titlesize": 10,
    }
)


def to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


def heatmap(
    ax,
    data,
    *,
    title="",
    xlabel="",
    ylabel="",
    xticks=None,
    yticks=None,
    diverging=False,
    vmin=None,
    vmax=None,
    lim=None,
    annotate=False,
    fmt="{:.2f}",
    colorbar=True,
):
    """Draw a matrix with row 0 at the top.

    Diverging maps are centred on 0 and span [-lim, lim] (default: the data's max |value|).
    """
    data = to_numpy(data)
    if diverging:
        lim = lim or np.nanmax(np.abs(data)) or 1.0
        norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        im = ax.imshow(data, cmap=DIVERGING, norm=norm, aspect="auto")
    else:
        im = ax.imshow(data, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    rows, cols = data.shape
    ax.set_xticks(range(cols), xticks if xticks is not None else range(cols))
    ax.set_yticks(range(rows), yticks if yticks is not None else range(rows))
    if cols > 16:
        ax.tick_params(axis="x", labelsize=6)
    if rows > 16:
        ax.tick_params(axis="y", labelsize=6)
    if annotate:
        for i in range(rows):
            for j in range(cols):
                if np.isnan(data[i, j]):
                    continue
                rgba = im.cmap(im.norm(data[i, j]))
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if dark else "black")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
