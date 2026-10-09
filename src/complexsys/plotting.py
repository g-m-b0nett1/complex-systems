"""Shared figure style: one look for every figure in the repository."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

# Categorical slots, used in this fixed order (validated for colour-vision deficiency).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
THEORY = INK  # theory curves: dark, dashed

# Two-state colormap for spin snapshots: minus = pale, plus = blue.
SPIN_CMAP = ListedColormap(["#e9e8e3", BLUE])


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )


def save(fig: plt.Figure, name: str, outdir: Path | str = "figures") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
