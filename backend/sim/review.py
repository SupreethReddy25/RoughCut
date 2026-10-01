"""Budgeted review sessions (PLAN 10, E8 "asking the right questions").

A creator with limited patience reviews only a fraction `budget` of the suggestions of each video. WHICH ones depends
on the review strategy:

- ``uncertainty``  the least confident suggestions first (the app's "Needs your eye" queue, PLAN 7.3);
- ``random``       a random subset;
- ``timeline``     from the start of the video until the budget is used up (what a person does without a queue).

Per video the policy first proposes the whole edit (proposals are committed one after the other, so the history
features follow the proposals), then the strategy picks what is reviewed. A reviewed disagreement is corrected with
the persona's `p_override`; everything else that is wrong stays wrong in the posted video. Feedback goes through the
same `Learner` as every other simulation. Two numbers per video: `correct` (the proposals that were right: what the
policy learned to do) and `final_correct` (decisions that are right in the video that is actually posted).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.policy.log_replay import Learner
from app.session import EditSession, Item
from sim.personas import Persona
from sim.synth import SynthVideo

STRATEGIES = ("uncertainty", "random", "timeline")


@dataclass
class BudgetVideo:
    index: int
    n: int
    correct: int  # proposals equal to the persona's ideal
    final_correct: int  # decisions right in the posted video (after the user's corrections)
    wrong: int
    wrong_fixed: int  # wrong proposals the review caught and corrected
    reviewed: int


def pick(strategy: str, confidences: list[float], k: int, rng: np.random.Generator) -> list[int]:
    """Indices of the `k` decisions that are reviewed."""
    n = len(confidences)
    k = max(0, min(k, n))
    if strategy == "uncertainty":
        return sorted(sorted(range(n), key=lambda i: (confidences[i], i))[:k])
    if strategy == "random":
        return sorted(int(i) for i in rng.choice(n, size=k, replace=False)) if k else []
    if strategy == "timeline":
        return list(range(k))
    raise ValueError(f"unknown review strategy {strategy!r}")


def run_budgeted(persona: Persona, learner: Learner, videos: list[SynthVideo], seed: int, budget: float,
                 strategy: str) -> list[BudgetVideo]:
    rng = np.random.default_rng(seed + 777)
    policy = learner.policy
    out: list[BudgetVideo] = []
    for vi, video in enumerate(videos, 1):
        p = persona.for_video(vi, video.video_type)
        p.prepare(video)
        sess = EditSession(video.uf, video.word_feats, video.video_type)
        rows: list[tuple[Item, str, float, str]] = []  # (item, proposal, confidence, ideal)
        for item in sess.items():
            choice = policy.choose(item.decision_type, item.ctx)
            rows.append((item, choice.action, choice.confidence, p.ideal_action(item)))
            sess.commit(item, choice.action)
        chosen = pick(strategy, [r[2] for r in rows], round(budget * len(rows)), rng)
        fixed = 0
        wrong = sum(action != ideal for _, action, _, ideal in rows)
        implicit: list[tuple[Item, str]] = []
        for i in chosen:
            item, action, _conf, ideal = rows[i]
            dt, arms = item.decision_type, (item.arms if item.decision_type == "emphasis" else None)
            feedback = ideal
            if rng.random() < p.noise:
                feedback = str(rng.choice([a for a in item.arms if a != ideal] or [ideal]))
            if feedback == action:
                learner.apply(dt, item.ctx, action, "accept", arms=arms)
            elif rng.random() < p.p_override:
                learner.apply(dt, item.ctx, action, "override", feedback, arms)
                fixed += action != ideal and feedback == ideal
            else:
                implicit.append((item, action))
        for item, action in implicit:  # seen, left untouched: finalize gives +0.5
            learner.apply(item.decision_type, item.ctx, action, "implicit",
                          arms=item.arms if item.decision_type == "emphasis" else None)
        n = len(rows)
        correct = n - wrong
        out.append(BudgetVideo(vi, n, correct, correct + fixed, wrong, fixed, len(chosen)))
    return out
