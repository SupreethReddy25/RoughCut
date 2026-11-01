"""The two conceptual diagrams of the paper, drawn as vector PDFs.

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

F = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "mathtext.fontset": "stix"})
PLAIN, HOT = "#eeeeee", "#fbd9b5"


def box(ax, x, y, w, h, text, hot=False, size=6.5, bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc=HOT if hot else PLAIN, ec="#333333", lw=.6))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 1.1 * size / 6.5 * 1.15, lines[0], ha="center", va="center", fontsize=size, fontweight="bold")
        ax.text(x + w / 2, y + (h - 2.0) / 2, "\n".join(lines[1:]), ha="center", va="center", fontsize=size, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.15)


def arrow(ax, a, b, text=None, off=(0, 0), dashed=False, rad=0.0, size=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=.7, color="#222222", ls="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="center", fontsize=size)


def canvas(w_in, h_in, w, h):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


# ---------------------------------------------------------------- Fig. 1: learning loop
fig, ax = canvas(3.45, 2.5, 69, 50)
box(ax, 2, 37, 40, 11, "Agent\nPosterior over weights $\\theta_a$\n(one model per creator)", hot=True, bold_first=True)
box(ax, 2, 15, 40, 11, "Environment\nRaw clip, analysis features,\nand the creator", bold_first=True)
box(ax, 49, 24, 18, 15, "Policy update\n$A$, $b$, and style\nposterior", hot=True, bold_first=True, size=6)
box(ax, 2, 1, 65, 8, "Inference: one posterior draw (or its mean) gives the first cut,\nexported as a 1080 × 1920 video", size=6)
arrow(ax, (20, 37), (20, 26), "Action $a_t$:\nsuggested edit", off=(-9, 0))
arrow(ax, (31, 26), (31, 37), "State $x_t$", off=(5, 0))
arrow(ax, (42, 20), (58, 24), "Reward $r_t$: accept,\nreject, or edit", off=(9, -7))
arrow(ax, (49, 34), (42, 40))
arrow(ax, (22, 15), (22, 9), dashed=True)
fig.savefig(F / "loop.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: full pipeline
fig, ax = canvas(7.1, 1.95, 142, 39)
W, H = 20.4, 13
xs = [1 + i * (W + 3.3) for i in range(6)]
top, bot = 24, 3
tops = [("Raw clip\n(English speech)", False), ("Transcription,\nalignment, disfluency\nrecovery", False),
        ("State extraction:\npauses, fillers, retakes,\nsemantics, face", False), ("Agent: LinTS per\ncreator with\nstyle prior", True),
        ("Posterior-sampling\nplanner (when a\ntarget is set)", True), ("Action applied to\nthe edit decision\nlist", False)]
bots = [("Off-policy\nevaluation", True), ("Event log with\ncontext and\npropensity", False), ("Reward mapping\n+1, −1, +0.5", False),
        ("Creator: accept,\nreject, or edit\nthe transcript", False), ("Review queue\nordered by\n1 − confidence", True),
        ("Export: 1080 × 1920,\ncaptions, −14 LUFS", False)]
for x, (t, hot) in zip(xs, tops):
    box(ax, x, top, W, H - 1, t, hot, size=6.3)
for x, (t, hot) in zip(xs, bots):
    box(ax, x, bot, W, H - 1, t, hot, size=6.3)
mid_t, mid_b = top + (H - 1) / 2, bot + (H - 1) / 2
for i in range(5):
    lab = {2: "$x_t$", 4: "$a_t$"}.get(i)
    arrow(ax, (xs[i] + W, mid_t), (xs[i + 1], mid_t), lab, off=(0, 1.8))
for i in (4, 3, 2, 1):  # queue -> creator -> reward -> log -> evaluation
    lab = {3: "$r_t$"}.get(i)
    arrow(ax, (xs[i], mid_b), (xs[i - 1] + W, mid_b), lab, off=(0, 1.8))
arrow(ax, (xs[5] + W / 2, top), (xs[5] + W / 2, bot + H - 1))
arrow(ax, (xs[5] + 2, top), (xs[4] + W - 2, bot + H - 1))
arrow(ax, (xs[2] + W - 4, bot + H - 1), (xs[3] + 4, top), "update", off=(-4.5, 0))
fig.savefig(F / "pipeline.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print("ok")
