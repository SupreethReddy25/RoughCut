"""E10: robustness to a wrong model. The other experiments use personas we wrote, whose rules a linear model on our
features can represent, and whose styles the priors were trained on. That favours our model family. Here the creators
are deliberately outside it, and none of them is in the training data of the prior:

    nonlinear   interaction and interval rules no single linear term can express (XOR-like keep/cut, pauses kept only
                inside a band of lengths, a filler kept only in some combinations)
    noisy       simple preferences, but 20 % of the feedback labels are wrong and only 30 % of disagreements get corrected
    arbitrary   decisions that depend on a hash of the unit: nothing to learn beyond the base rates (the floor)

The result is reported whichever way it goes: acceptance of LinTS against the rule baseline and the frozen generic
prior, first videos and last videos. Only keep/cut, pause and filler decisions count: these personas never zoom or
highlight, so including those would mostly measure "learned to stop zooming". `constant` is the best a policy can do
that ignores the context and picks one fixed arm per decision type, the bar for having learned anything specific.

    python -m sim.experiments.e10 [--seeds 8] [--videos 8]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.experiments import plots
from sim.experiments.common import ci95, generic_prior, make_learner, run_parallel, summarize, videos_for
from sim.personas import Persona
from sim.session import run_user

ALGOS = {"rule": "#999999", "global": "#E69F00", "lints": "#D55E00"}
LABEL = {"rule": "Rule (simple editor)", "global": "Generic prior, frozen", "lints": "LinTS (ours)"}


@dataclass
class Nonlinear(Persona):
    name: str = "nonlinear"

    def phrase(self, c, item) -> str:
        if c.get("is_earlier_take", 0.0) >= 0.5:
            return "CUT"
        long, central = c.get("duration", 0.0) > 2.5, c.get("topic_centrality", 0.0) > 0.45
        return "KEEP" if long == central else "CUT"  # XOR: long and central, or short and off-topic

    def pause(self, c) -> str:
        g = c.get("gap_seconds", 0.0)
        return "KEEP" if 0.5 <= g < 0.8 else ("TIGHT_15" if g < 1.0 else "REMOVE")

    def filler(self, c) -> str:
        return "KEEP" if (c.get("pos_start", 0.0) > 0.5) != (c.get("loudness_z", 0.0) > 0.0) else "REMOVE"

    def zoom(self, c) -> str:
        return "Z115" if (c.get("loudness_z", 0.0) > 0.5) != (c.get("duration", 0.0) > 3.0) else "NONE"


@dataclass
class Noisy(Persona):
    name: str = "noisy"
    noise: float = 0.2
    p_override: float = 0.3

    def pause(self, c) -> str:
        return "TIGHT_40" if c.get("gap_seconds", 0.0) >= 0.5 else "KEEP"


@dataclass
class Arbitrary(Persona):
    name: str = "arbitrary"

    def _h(self, c, n: int) -> int:
        return int(abs(c.get("position", 0.0) * 9973 + c.get("duration", c.get("gap_seconds", 0.0)) * 7919)) % n

    def phrase(self, c, item) -> str:
        return "CUT" if self._h(c, 4) == 0 else "KEEP"

    def pause(self, c) -> str:
        return ("KEEP", "TIGHT_40", "TIGHT_15", "REMOVE")[self._h(c, 4)]

    def filler(self, c) -> str:
        return ("KEEP", "REMOVE")[self._h(c, 2)]


PERSONAS = {"nonlinear": Nonlinear, "noisy": Noisy, "arbitrary": Arbitrary}
HARD = ("phrase", "pause", "filler")


def hard_acc(results: list[dict]) -> np.ndarray:
    return np.array([[sum(v["correct"][d] for d in HARD) / max(sum(v["n"][d] for d in HARD), 1) for v in r["videos"]]
                     for r in results])


def best_constant(persona: Persona, seed: int, n_videos: int) -> float:
    """Acceptance of the best fixed arm per decision type, on the same videos (noise-free ideal actions)."""
    from collections import Counter

    from app.session import EditSession

    counts: dict[str, Counter] = {d: Counter() for d in HARD}
    for k, video in enumerate(videos_for(seed, n_videos), 1):
        p = persona.for_video(k, video.video_type)
        p.prepare(video)
        sess = EditSession(video.uf, video.word_feats, video.video_type)
        for item in sess.items():
            ideal = p.ideal_action(item)
            if item.decision_type in HARD:
                counts[item.decision_type][ideal] += 1
            sess.commit(item, ideal)
    return sum(c.most_common(1)[0][1] for c in counts.values() if c) / max(sum(sum(c.values()) for c in counts.values()), 1)


def e10_task(task: dict) -> dict:
    learner = make_learner(task["algo"], generic_prior(), task["seed"], 0.995)
    run = run_user(PERSONAS[task["persona"]](), learner, videos_for(task["seed"], task["videos"]), task["seed"])
    return {**task, "videos": summarize(run), "constant": best_constant(PERSONAS[task["persona"]](), task["seed"],
                                                                        task["videos"])}


def run(seeds: int = 8, n_videos: int = 8, workers: int | None = None) -> list[dict]:
    tasks = [{"persona": p, "algo": a, "seed": s, "videos": n_videos}
             for p in PERSONAS for a in ALGOS for s in range(1, seeds + 1)]
    return run_parallel(e10_task, tasks, workers)


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    table: dict = {}
    fig, axes = plots.plt.subplots(1, len(PERSONAS), figsize=(14, 4.2), sharey=True)
    for ax, p in zip(axes, PERSONAS, strict=True):
        table[p] = {}
        for a, color in ALGOS.items():
            m, h = ci95(hard_acc([r for r in results if r["persona"] == p and r["algo"] == a]))
            table[p][a] = {"mean": m.tolist(), "ci95": h.tolist()}
            plots.curve(ax, np.arange(1, len(m) + 1), m, h, LABEL[a], color)
        table[p]["constant"] = float(np.mean([r["constant"] for r in results if r["persona"] == p]))
        ax.axhline(table[p]["constant"], color="black", lw=1, ls="--", label="best fixed arm per type")
        plots.style_axes(ax, "acceptance (keep/cut, pause, filler)" if p == "nonlinear" else None, "video")
        ax.set_title(p)
    axes[0].legend(loc="lower right")
    fig.suptitle("E10: creators outside the model family and outside the prior's training data (mean ± 95% CI)")
    plots.save(fig, "e10_robustness.png", out_dir)
    summary = {"table": table}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e10_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E10: robustness to a misspecified model")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    s = report(run(a.seeds, a.videos, a.workers))
    for p, t in s["table"].items():
        print(p, "best fixed arm", round(t["constant"], 3),
              {k: (round(v["mean"][0], 3), round(v["mean"][-1], 3), round(v["ci95"][-1], 3)) for k, v in t.items()
               if k != "constant"})


if __name__ == "__main__":
    main()
