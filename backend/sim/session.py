"""Simulated editing sessions (PLAN 9): a persona reviews a policy's suggestions and gives feedback.

For every decision (left to right, via the shared `EditSession`): the policy proposes; the persona compares with its
hidden ideal. Reviewed + agree -> accept. Reviewed + disagree -> with prob `p_override` correct it (half of the
corrections happen as a *manual edit* - delete/restore words, set pause, ... - which goes through the real
classification `reward.classify_manual_edit`), otherwise leave it untouched (implicit +0.5 at finalize, since it
was seen). Not reviewed -> never seen -> no feedback. All updates go through `Learner`, the same path the API uses.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.edl.build import PAD, PAUSE_TARGET
from app.policy.log_replay import Learner
from app.policy.reward import ManualEdit, classify_manual_edit
from app.schemas import Event, Suggestion
from app.session import EditSession, Item, make_planner, make_suggestion, planned_choice
from sim.personas import Persona
from sim.synth import SynthVideo

DECISION_TYPES = ("phrase", "pause", "filler", "zoom", "emphasis")
P_MANUAL = 0.5  # share of corrections made as a manual edit instead of in the suggestion panel


@dataclass
class VideoResult:
    index: int
    video_type: str
    minutes: float
    n: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    corrections: int = 0  # override / reject events
    manual_edits: int = 0  # of which made as manual edits
    accepts: int = 0
    implicits: int = 0
    out_seconds: float = 0.0

    @property
    def acceptance(self) -> float:
        n = sum(self.n.values())
        return sum(self.correct.values()) / n if n else float("nan")

    def acceptance_of(self, dt: str) -> float:
        return self.correct[dt] / self.n[dt] if self.n[dt] else float("nan")


@dataclass
class UserRun:
    persona: str
    policy: str
    seed: int
    videos: list[VideoResult]
    suggestions: list[Suggestion] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    # simulator ground truth per logged suggestion (for off-policy evaluation, E9): the persona's ideal action
    # and its feedback parameters at that moment
    truth: list[dict] = field(default_factory=list)


def _manual_edit(item: Item, final: str, rng: np.random.Generator) -> ManualEdit:
    """The manual edit a person would make to turn the proposal of `item` into `final`."""
    dt = item.decision_type
    if dt == "phrase":
        return ManualEdit("delete_words" if final == "CUT" else "restore_words", word_ids=list(item.word_ids))
    if dt == "filler":
        return ManualEdit("delete_words" if final == "REMOVE" else "restore_words", word_ids=list(item.word_ids))
    if dt == "pause":
        gap = item.unit.end - item.unit.start
        if rng.random() < 0.5:
            return ManualEdit("set_pause", unit_id=item.target_id, option=final)
        left = gap if final == "KEEP" else min(gap, 2 * PAD + PAUSE_TARGET[final])
        return ManualEdit("drag_pause", unit_id=item.target_id, out_gap=left)
    if dt == "zoom":
        return ManualEdit("set_zoom", unit_id=item.target_id, option=final)
    word = None if final == "NONE" else item.candidate_word_ids[int(final.split("_")[1])]
    return ManualEdit("set_emphasis", line_idx=item.target_id, word_id=word, line_word_ids=list(item.line_word_ids),
                      candidate_word_ids=list(item.candidate_word_ids))


def run_user(persona: Persona, learner: Learner, videos: list[SynthVideo], seed: int, keep_log: bool = False,
             p_manual: float = P_MANUAL, plan: bool = False) -> UserRun:
    """One simulated creator editing `videos` in order with `learner.policy` suggesting. Learns online.
    `plan`: when the persona has a target duration, phrase/pause suggestions come from the episode planner."""
    rng = np.random.default_rng(seed + 12345)
    policy = learner.policy
    run = UserRun(persona.name, policy.name, seed, [])
    for vi, video in enumerate(videos, 1):
        p = persona.for_video(vi, video.video_type)
        p.prepare(video)
        sess = EditSession(video.uf, video.word_feats, video.video_type)
        planner = make_planner(video.uf, video.video_type, policy, p.target_duration,
                               seed=seed * 1000 + vi) if plan else None
        res = VideoResult(vi, video.video_type, video.minutes)
        implicit: list[tuple[Item, str, int | None]] = []
        for item in sess.items():
            dt, ctx = item.decision_type, item.ctx
            planned = planned_choice(planner, item, sess)
            choice = planned[0] if planned is not None else policy.choose(dt, ctx)
            ideal = p.ideal_action(item)
            res.n[dt] += 1
            res.correct[dt] += choice.action == ideal
            sid = None
            if keep_log:
                sid = len(run.suggestions)
                sug = make_suggestion(item, choice, policy, video.uf.video_id)
                sug.id = sid
                run.suggestions.append(sug)
                run.truth.append({"ideal": ideal, "noise": p.noise, "p_override": p.p_override,
                                  "review": p.review_budget, "video": vi})
            final = choice.action
            if rng.random() < p.review_budget:  # reviewed => seen
                feedback = ideal
                if rng.random() < p.noise:
                    feedback = str(rng.choice([a for a in item.arms if a != ideal] or [ideal]))
                arms = item.arms if dt == "emphasis" else None
                if feedback == choice.action:
                    learner.apply(dt, ctx, choice.action, "accept", arms=arms)
                    res.accepts += 1
                    kind, via = "accept", None
                elif rng.random() < p.p_override:
                    via = "panel"
                    if rng.random() < p_manual:
                        maps = classify_manual_edit(_manual_edit(item, feedback, rng), video.uf,
                                                    sess.preview(item, choice.action))
                        ov = [m for m in maps if m.kind == "override" and m.decision_type == dt
                              and m.target_id == item.target_id and m.final_action == feedback]
                        if len(maps) == 1 and ov:  # otherwise the edit means something else: use the panel
                            via = "manual"
                            res.manual_edits += 1
                    learner.apply(dt, ctx, choice.action, "override", feedback, arms, manual=via == "manual")
                    res.corrections += 1
                    final, kind = feedback, "override"
                else:
                    implicit.append((item, choice.action, sid))
                    kind, via = None, None
                if keep_log and kind:
                    run.events.append(Event(id=len(run.events), user_id=0, video_id=video.uf.video_id, kind=kind,  # type: ignore[arg-type]
                                            suggestion_id=sid,
                                            payload={"final_action": final, **({"via": "manual"} if via == "manual" else {})}))
            sess.commit(item, final)
        for item, proposed, sid in implicit:  # finalize: untouched + seen -> +0.5
            learner.apply(item.decision_type, item.ctx, proposed, "implicit",
                          arms=item.arms if item.decision_type == "emphasis" else None)
            res.implicits += 1
            if keep_log:
                run.events.append(Event(id=len(run.events), user_id=0, video_id=video.uf.video_id, kind="implicit",
                                        suggestion_id=sid, payload={}))
        res.out_seconds = sess._out_t
        run.videos.append(res)
    return run


class GreedyView:
    """Deterministic posterior-mean view of a policy (no exploration) for evaluating what was learned."""

    def __init__(self, policy):
        self.policy = policy
        self.name = policy.name

    def choose(self, decision_type, ctx, offsets=None):
        scores = self.policy.mean_scores(decision_type, ctx)
        return max(scores, key=lambda a: scores[a])


def evaluate(choose_fn, persona: Persona, videos: list[SynthVideo], start_index: int = 1) -> dict[str, float]:
    """Acceptance of a frozen chooser (decision_type, ctx) -> action against `persona`'s ideal, per type + overall.
    The chooser's own proposals are committed, so history features follow what it would really produce."""
    n: dict[str, int] = defaultdict(int)
    ok: dict[str, int] = defaultdict(int)
    for vi, video in enumerate(videos, start_index):
        p = persona.for_video(vi, video.video_type)
        p.prepare(video)
        sess = EditSession(video.uf, video.word_feats, video.video_type)
        for item in sess.items():
            a = choose_fn(item.decision_type, item.ctx)
            n[item.decision_type] += 1
            ok[item.decision_type] += a == p.ideal_action(item)
            sess.commit(item, a)
    out = {dt: ok[dt] / n[dt] for dt in n}
    out["overall"] = sum(ok.values()) / max(sum(n.values()), 1)
    return out
