"""E11: is the visual rhythm a sequential problem worth RLSVI, or would a bandit on the same features do as well?

Simulated creators with rhythm preferences that depend on what happened before:

    steady     one modest punch-in after a quiet stretch (at least 6 s), never two in a row
    buildup    saves a strong beat for important phrases: only after 8 quiet seconds, otherwise nothing
    busy       a beat every 3 s, and never the same kind twice in a row
    minimal    almost nothing: one text pop-up on a really central phrase after 15 quiet seconds

Each edits 8 videos with feedback on every beat (noise 5 %, corrections 60 %) and rates the whole video at the end by how
close its beats per minute are to what they like. Three agents, with exactly the same features:

    rule     a fixed 1.15x punch-in every 6 s
    bandit   the same linear Thompson model with gamma = 0: maximises each beat on its own, ignores the rating
    rlsvi    gamma = 0.9, whole-video exploration, and the rating as a delayed reward

Metrics on 4 held-out videos after each training video, acting greedily: share of beats equal to the creator's ideal,
and the whole-video rating. The creators are written by us, like the personas of the other experiments, and every state
they react to is visible to the bandit too, so a tie would be a fair result: report whatever comes out.

    python -m sim.experiments.e11 [--seeds 6] [--videos 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.rl.rhythm import GAMMA, Episode, RhythmAgent, Step
from app.session import EditSession
from sim.experiments import plots
from sim.experiments.common import ci95, run_parallel, videos_for

N_TEST = 4
NOISE, P_OVERRIDE = 0.05, 0.6
COLOR = {"rule": "#999999", "bandit": "#0072B2", "rlsvi": "#D55E00"}


def ideal(name: str, c: dict[str, float]) -> str:
    since, last = c["hist_since_visual"], int(c["hist_last_visual"])
    central = c.get("topic_centrality", 0.0)
    if name == "steady":
        return "Z115" if since >= 6 and last != 1 or since >= 12 else "NONE"
    if name == "buildup":
        return "Z130" if since >= 8 and central >= 0.45 else "NONE"
    if name == "busy":
        if since < 3:
            return "NONE"
        return {0: "Z115", 1: "TEXT_POP", 2: "SFX", 3: "Z115"}[last]
    return "TEXT_POP" if since >= 15 and central >= 0.55 else "NONE"


LIKES_PER_MIN = {"steady": 6.0, "buildup": 4.0, "busy": 14.0, "minimal": 1.5}
PERSONAS = tuple(LIKES_PER_MIN)


def _rule(c: dict[str, float]) -> str:
    return "Z115" if c["hist_since_visual"] >= 6 else "NONE"


def _walk(video, choose):
    """One video: `choose(item)` -> (action to commit) for rhythm items; every other decision stays neutral."""
    sess = EditSession(video.uf, video.word_feats, video.video_type, agents=True)
    for item in sess.items():
        if item.decision_type == "rhythm":
            sess.commit(item, choose(item))
        elif item.decision_type == "phrase":
            sess.commit(item, "CUT" if item.ctx.get("is_earlier_take", 0.0) >= 0.5 else "KEEP")
        elif item.decision_type == "pause":
            sess.commit(item, "TIGHT_40")
        else:
            sess.commit(item, item.arms[0] if item.decision_type != "emphasis" else "NONE")
    return sess


def _rating(name: str, beats: int, out_seconds: float) -> float:
    per_min = beats / max(out_seconds / 60.0, 1e-6)
    miss = abs(per_min - LIKES_PER_MIN[name]) / LIKES_PER_MIN[name]
    return 1.0 if miss < 0.25 else 0.0 if miss < 0.6 else -1.0


def _eval(name: str, tests, act) -> dict[str, float]:
    ok, n, ratings = 0, 0, []
    for video in tests:
        beats = [0]

        def choose(item):
            a = act(item.ctx)
            nonlocal ok, n
            ok += a == ideal(name, item.ctx)
            n += 1
            beats[0] += a != "NONE"
            return a

        sess = _walk(video, choose)
        ratings.append(_rating(name, beats[0], sess._out_t))
    return {"acc": ok / max(n, 1), "rating": float(np.mean(ratings))}


def e11_task(task: dict) -> dict:
    name, seed, algo = task["persona"], task["seed"], task["algo"]
    tests = videos_for(seed, N_TEST, offset=500)
    rng = np.random.default_rng(seed * 31 + 7)
    if algo == "rule":
        curve = [_eval(name, tests, _rule)] * (task["videos"] + 1)
        return {**task, "curve": curve}
    kw = task.get("kw", {}) if algo == "rlsvi" else {}
    agent = RhythmAgent(seed=seed, gamma=0.0 if algo == "bandit" else kw.get("gamma", GAMMA), n_draws=0,
                        **{k: v for k, v in kw.items() if k != "gamma"})

    def mean_act(ctx):
        return agent.mean_action(ctx)

    agent.begin_video()
    curve = [_eval(name, tests, mean_act)]
    for video in videos_for(seed, task["videos"]):
        agent.begin_video()
        ep = Episode()
        beats = [0]

        def choose(item):
            a = agent.choose(item.ctx).action
            want = ideal(name, item.ctx)
            if rng.random() < NOISE:
                want = str(rng.choice([x for x in item.arms if x != want]))
            step = Step(dict(item.ctx), a)
            final = a
            if want == a:
                step.reward = 1.0
            elif rng.random() < P_OVERRIDE:
                step.reward, step.final = -1.0, want
                final = want
            else:
                step.reward = 0.5  # left alone
            ep.steps.append(step)
            beats[0] += final != "NONE"
            return final

        sess = _walk(video, choose)
        if algo == "rlsvi":
            ep.rating = _rating(name, beats[0], sess._out_t)
        agent.add_episode(ep)
        agent.begin_video()
        curve.append(_eval(name, tests, mean_act))
    return {**task, "curve": curve}


def run(seeds: int = 6, n_videos: int = 8, workers: int | None = None, seed_base: int = 0) -> list[dict]:
    tasks = [{"persona": p, "algo": a, "seed": seed_base + s, "videos": n_videos} for p in PERSONAS for a in COLOR
             for s in range(1, seeds + 1)]
    return run_parallel(e11_task, tasks, workers)


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    table: dict = {}
    fig, axes = plots.plt.subplots(2, len(PERSONAS), figsize=(15, 6.5), sharex=True)
    for j, p in enumerate(PERSONAS):
        table[p] = {}
        for algo, color in COLOR.items():
            rs = [r for r in results if r["persona"] == p and r["algo"] == algo]
            table[p][algo] = {}
            for i, key in enumerate(("acc", "rating")):
                m, h = ci95(np.array([[c[key] for c in r["curve"]] for r in rs]))
                table[p][algo][key] = {"mean": m.tolist(), "ci95": h.tolist()}
                plots.curve(axes[i, j], np.arange(len(m)), m, h, algo, color)
        axes[0, j].set_title(p)
        plots.style_axes(axes[0, j], "beats equal to the creator's ideal" if j == 0 else None, None)
        plots.style_axes(axes[1, j], "whole-video rating" if j == 0 else None, "videos edited")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("E11: visual rhythm, RLSVI vs a bandit on the same features (held-out videos, mean ± 95% CI)")
    plots.save(fig, "e11_rhythm.png", out_dir)
    summary = {"table": table, "held_out_videos": N_TEST}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e11_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E11: visual rhythm, RLSVI vs bandit")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=100, help="settings were tuned on seeds 1-6: report on others")
    a = ap.parse_args()
    s = report(run(a.seeds, a.videos, a.workers, a.seed_base))
    for p, t in s["table"].items():
        print(p, {k: (round(v["acc"]["mean"][-1], 3), round(v["rating"]["mean"][-1], 2)) for k, v in t.items()})
    for algo in COLOR:
        print(f"{algo:7} acc {np.mean([s['table'][p][algo]['acc']['mean'][-1] for p in PERSONAS]):.3f}  "
              f"rating {np.mean([s['table'][p][algo]['rating']['mean'][-1] for p in PERSONAS]):+.2f}")


if __name__ == "__main__":
    main()
