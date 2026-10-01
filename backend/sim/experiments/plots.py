"""Matplotlib helpers shared by the experiments (Okabe-Ito colour-blind-safe palette, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"rule": "#E69F00", "global": "#999999", "eps": "#56B4E9", "linucb": "#009E73", "lints": "#D55E00"}
FIG_DIR = Path(__file__).resolve().parents[3] / "reports" / "figures"


def curve(ax, x, mean, half, label: str, color: str, lw: float = 1.8, ls: str = "-") -> None:
    ax.plot(x, mean, color=color, lw=lw, ls=ls, label=label, marker="o", ms=3)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.15, lw=0)


def style_axes(ax, ylabel: str | None = None, xlabel: str | None = None, ylim=None) -> None:
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(*ylim)


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    out = (out_dir or FIG_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def heatmap(ax, M: np.ndarray, rows: list[str], cols: list[str], title: str = "") -> None:
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] < 0.6 else "black",
                    fontweight="bold" if i == j else None, fontsize=9)
    ax.set_title(title)
    return im
