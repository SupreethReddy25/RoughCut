"""E5 (PLAN 7.6, 10): the sequential layer. Can the agent fit an edit to a target duration while still editing the
way the creator likes, and does that need planning over the whole video (an MDP) or is a per-decision bandit enough?

Creators with a target length (the Minimalist persona keeps its most central phrases until the target is reached and
removes every pause; targets 20 / 30 / 45 s) edit 6 videos while learning online. After each video the learned
policy is FROZEN and applied, without any corrections, to 4 held-out videos (what autopilot would post):

    bandit       LinTS posterior mean, one decision at a time (no notion of the target)
    bandit+plan  the same learned reward model, but phrase/pause decisions come from the finite-horizon MDP solved by
                 backward induction with the target as terminal cost (app/rl/plan.py)

Training itself uses posterior sampling (PSRL) for the planner arm and Thompson sampling for the bandit arm.
Metrics: |output length - target| / target, and acceptance (share of decisions equal to the creator's ideal).

    python -m sim.experiments.e5 [--seeds 8] [--videos 6]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sim.experiments import plots
from sim.experiments.common import ci95, generic_prior, make_learner, run_parallel, videos_for
from sim.personas import make_persona
from sim.session import GreedyView, run_user

TARGETS = (20.0, 30.0, 45.0)
N_TEST = 4
ARMS = {"bandit": "#0072B2", "bandit+plan": "#D55E00"}


def frozen_eval(policy, persona, videos, plan: bool) -> dict[str, float]:
    """Apply the posterior mean (no exploration, no corrections) to `videos`; duration error + acceptance."""
    from app.session import EditSession, make_planner, planned_choice

    errs, ok, n, ok_ph, n_ph = [], 0, 0, 0, 0
    greedy = GreedyView(policy)
    for vi, video in enumerate(videos, 100):
        p = persona.for_video(vi, video.video_type)
        p.prepare(video)
        sess = EditSession(video.uf, video.word_feats, video.video_type)
        planner = make_planner(video.uf, video.video_type, policy, p.target_duration, greedy=True) if plan else None
        for item in sess.items():
            planned = planned_choice(planner, item, sess)
            a = planned[0].action if planned is not None else greedy.choose(item.decision_type, item.ctx)
            good = a == p.ideal_action(item)
            ok += good
            n += 1
            if item.decision_type == "phrase":
                ok_ph += good
                n_ph += 1
            sess.commit(item, a)
        # measured on the real edit (the EDL), not on the planner's step model
        errs.append(abs(sess.build_edl().duration - p.target_duration) / p.target_duration)
    return {"dur_err": float(np.mean(errs)), "acc": ok / max(n, 1), "acc_phrase": ok_ph / max(n_ph, 1)}


def e5_task(task: dict) -> dict:
    seed, target, plan = task["seed"], task["target"], task["arm"] == "bandit+plan"
    persona = make_persona("minimalist", target_duration=target)
    learner = make_learner("lints", generic_prior(("minimalist",)), seed, 0.995)
    test = videos_for(seed, N_TEST, offset=500)
    curve = [frozen_eval(learner.policy, persona, test, plan)]  # before any feedback (the prior)
    for vi, video in enumerate(videos_for(seed, task["videos"]), 1):
        run_user(persona, learner, [video], seed * 100 + vi, plan=plan)
        curve.append(frozen_eval(learner.policy, persona, test, plan))
    return {**task, "curve": curve}


def run(seeds: int = 8, n_videos: int = 6, workers: int | None = None) -> list[dict]:
    tasks = [{"seed": s, "target": t, "arm": a, "videos": n_videos}
             for s in range(1, seeds + 1) for t in TARGETS for a in ARMS]
    return run_parallel(e5_task, tasks, workers)


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    table: dict = defaultdict(dict)
    for arm in ARMS:
        rs = [r for r in results if r["arm"] == arm]
        for key in ("dur_err", "acc", "acc_phrase"):
            m, h = ci95(np.array([[c[key] for c in r["curve"]] for r in rs]))
            table[arm][key] = {"mean": m.tolist(), "ci95": h.tolist()}
        table[arm]["by_target"] = {str(t): float(np.mean([r["curve"][-1]["dur_err"] for r in rs if r["target"] == t]))
                                   for t in TARGETS}
    summary = {"targets": list(TARGETS), "held_out_videos": N_TEST, "table": table}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e5_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    fig, axes = plots.plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(results[0]["curve"]))
    for ax, key, title in zip(axes, ("dur_err", "acc", "acc_phrase"),
                              ("|length - target| / target  (lower is better)", "acceptance, all decisions",
                               "acceptance, keep/cut decisions"), strict=True):
        for arm, color in ARMS.items():
            plots.curve(ax, x, np.array(table[arm][key]["mean"]), np.array(table[arm][key]["ci95"]), arm, color)
        plots.style_axes(ax, title, "videos edited (0 = prior only)")
        ax.set_xticks(x)
    axes[0].legend(loc="upper right")
    fig.suptitle("E5: fitting a target length, per-decision bandit vs planning over the whole edit "
                 "(held-out videos, mean ± 95% CI)")
    plots.save(fig, "e5_target_duration.png", out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E5: target duration, bandit vs MDP planner")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    s = report(run(a.seeds, a.videos, a.workers))
    for arm in ARMS:
        t = s["table"][arm]
        print(f"{arm:12} length error {t['dur_err']['mean'][0]:.2f} -> {t['dur_err']['mean'][-1]:.2f}   "
              f"acceptance {t['acc']['mean'][0]:.3f} -> {t['acc']['mean'][-1]:.3f}   "
              f"keep/cut {t['acc_phrase']['mean'][0]:.3f} -> {t['acc_phrase']['mean'][-1]:.3f}   "
              f"by target {t['by_target']}")


if __name__ == "__main__":
    main()
