"""Left-to-right suggestion generation (PLAN 5, 6.2), shared by the product and the simulator.

Order: per position, pause -> phrase -> (if kept) its filler tokens -> zoom.  After all units the caption lines are
built from the kept words (using the decisions made so far) and each line gets a D5 emphasis decision.

`EditSession.items()` is a generator that the consumer drives:  for each yielded Item it decides an action
(policy proposal, or what a simulated / real user ends up with) and calls `commit(item, action)`. The history
features (hist_*) are computed from the committed actions, which is why decisions are sequential.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.edl.build import (
    PAD,
    PAUSE_TARGET,
    BuildOptions,
    Decisions,
    build_edl_with_lines,
)
from app.policy import context
from app.policy.base import Choice, Policy
from app.schemas import EDL, Suggestion, Unit, UnitsFile, Word

FEATURE_VERSION = context.CONTEXT_VERSION
ZOOM_NEVER_S = 30.0  # seconds-since-a-visual-change before the first one


@dataclass
class Item:
    """One decision presented to the policy (and, in the simulator, to a persona)."""

    decision_type: str
    target_id: int  # unit id (phrase / pause / zoom / filler) or caption line index (emphasis)
    ctx: dict[str, float]  # context_raw
    arms: list[str]
    unit: Unit | None = None
    word_ids: list[int] = field(default_factory=list)  # words the decision is about
    candidate_word_ids: list[int] = field(default_factory=list)  # emphasis: candidate k -> word id
    line_word_ids: list[int] = field(default_factory=list)  # emphasis: all words of the caption line


@dataclass
class SuggestionSet:
    suggestions: list[Suggestion]
    items: list[Item]
    decisions: Decisions
    skipped_fillers: list[int]  # filler units with no clean gap: default KEEP, no suggestion (PLAN 12)
    mean_best: list[str] = field(default_factory=list)  # per suggestion: the posterior-mean argmax when it was made


# A new decision type starts with no learned weights, so ties would be broken at random. These keep the plain choice
# unless the creator's feedback says otherwise.
DEFAULT_OFFSETS = {"pace": {"S100": 0.8}, "music_timing": {"FROM_START": 0.4}}


class EditSession:
    def __init__(self, uf: UnitsFile, word_feats: dict[int, dict[str, float]] | None = None,
                 video_type: str | None = None, agents: bool = False):
        self.uf = uf
        self.agents = agents  # the product's decision set: rhythm + pace per phrase, hook + music timing per video
        self._last_visual_t: float | None = None
        self._last_visual = 0
        self._streak = 0
        self.word_feats = word_feats or {}
        self.video_type = video_type or uf.video_type
        self.decisions = Decisions()
        self.skipped_fillers: list[int] = []
        self.words_by_id = {w.id: w for w in uf.words}
        self._fillers: dict[int, list[Unit]] = {}
        for u in uf.units:
            if u.kind == "filler":
                self._fillers.setdefault(u.parent_unit_id, []).append(u)
        self._words_of: dict[int, list[Word]] = {}
        for w in uf.words:
            self._words_of.setdefault(w.unit_id, []).append(w)
        speech = [u for u in uf.units if u.kind == "speech"]
        self._t0 = speech[0].start if speech else 0.0
        # running history (approximate output time: kept speech + retained pauses)
        self._src_t = self._t0
        self._out_t = 0.0
        self._since_cut = 0
        self._last_zoom_end: float | None = None
        self._prev_zoomed = False
        self._phrase_out_start = 0.0

    # ------------------------------------------------------------------ history features
    def _hist(self) -> dict[str, float]:
        elapsed = self._src_t - self._t0
        since_zoom = (min(self._phrase_out_start - self._last_zoom_end, context.ZOOM_NEVER)
                      if self._last_zoom_end is not None else context.ZOOM_NEVER)
        return {
            "hist_output_ratio": min(self._out_t / elapsed, 1.5) if elapsed > 1e-6 else 1.0,
            "hist_since_last_cut": float(self._since_cut),
            "hist_since_last_zoom": max(since_zoom, 0.0),
            "hist_prev_zoomed": float(self._prev_zoomed),
        }

    def _item(self, dt: str, u: Unit, word_ids: list[int]) -> Item:
        hist = self._hist()
        if dt == "rhythm":
            since = (self._phrase_out_start - self._last_visual_t if self._last_visual_t is not None else ZOOM_NEVER_S)
            hist.update(hist_since_visual=min(max(since, 0.0), ZOOM_NEVER_S), hist_last_visual=float(self._last_visual),
                        hist_streak=float(self._streak))
        ctx = context.build_context(u.features, hist, self.video_type)
        return Item(dt, u.id, ctx, context.arms_for(dt), u, word_ids)

    # ------------------------------------------------------------------ generation
    def items(self) -> Iterator[Item]:
        uf = self.uf
        pause_before: dict[int, Unit] = {}
        speech: list[Unit] = []
        for u in uf.units:
            if u.kind == "speech":
                speech.append(u)
            elif u.kind == "pause":
                pause_before[len(speech)] = u  # the pause that follows speech[len-1] precedes speech[len]
        for k, su in enumerate(speech):
            p = pause_before.get(k)
            if p is not None and k > 0:
                yield self._item("pause", p, [])
            yield self._item("phrase", su, [w.id for w in self._words_of.get(su.id, [])])
            if self.decisions.action(su.id) == "CUT":
                continue
            for fu in self._fillers.get(su.id, []):
                if fu.features.get("clean_gap", 1.0) < 1.0:
                    self.skipped_fillers.append(fu.id)
                    continue
                yield self._item("filler", fu, [w.id for w in self._words_of.get(su.id, [])
                                                if w.is_filler and fu.start - 1e-3 <= w.start and w.end <= fu.end + 1e-3])
            if self.agents:
                ids = [w.id for w in self._words_of.get(su.id, [])]
                yield self._item("rhythm", su, ids)
                yield self._item("pace", su, ids)
            else:
                yield self._item("zoom", su, [w.id for w in self._words_of.get(su.id, [])])
        yield from self._emphasis_items()
        if self.agents:
            yield from self._video_items()

    def _emphasis_items(self) -> Iterator[Item]:
        _, line_ids = build_edl_with_lines(self.uf, self.decisions)
        for idx, ids in enumerate(line_ids):
            built = context.build_emphasis_context([self.words_by_id[i] for i in ids], self.word_feats,
                                                   self.video_type)
            if built is None:
                continue
            ctx, cand = built
            yield Item("emphasis", idx, ctx, context.arms_for("emphasis", ctx), None, list(cand), cand, list(ids))

    def _video_items(self) -> Iterator[Item]:
        from app.rl import hook

        cands = hook.candidates(self.uf)
        if cands:
            ctx = hook.context_of(cands)
            yield Item("hook", 0, ctx, hook.arms_of(ctx))
        first_end = next((w.end for w in self.uf.words if w.text_display.strip().endswith((".", "?", "!"))), 0.0)
        ctx = {**context.type_onehot(self.video_type), "bias": 1.0, "video_minutes": self.uf.duration / 60.0,
               "first_sentence_seconds": first_end - self._t0}
        yield Item("music_timing", 0, ctx, list(context.ARMS["music_timing"]))

    # ------------------------------------------------------------------ commit
    def commit(self, item: Item, action: str) -> None:
        """Record the action that ends up in the edit for `item` and advance the running history."""
        if action not in item.arms:
            raise ValueError(f"{action!r} is not an arm of this {item.decision_type} decision: {item.arms}")
        u, dt = item.unit, item.decision_type
        if dt == "pause":
            self.decisions.actions[item.target_id] = action
            gap = u.end - u.start
            self._out_t += gap if action == "KEEP" else min(gap, 2 * PAD + PAUSE_TARGET[action])
            self._src_t = u.end
        elif dt == "phrase":
            self.decisions.actions[item.target_id] = action
            self._phrase_out_start = self._out_t
            self._src_t = u.end
            if action == "CUT":
                self._since_cut = 0
                self._prev_zoomed = False
            else:
                self._out_t += u.end - u.start
                self._since_cut += 1
        elif dt == "filler":
            self.decisions.actions[item.target_id] = action
            if action == "REMOVE":
                self._out_t -= u.features.get("token_duration", 0.0)
        elif dt == "rhythm":
            self.decisions.rhythm[item.target_id] = action
            if action == "NONE":
                self._streak = 0
            else:
                self._streak += 1
                self._last_visual_t = self._phrase_out_start
                self._last_visual = {"Z115": 1, "Z130": 1, "TEXT_POP": 2, "SFX": 3}[action]
        elif dt == "pace":
            self.decisions.pace[item.target_id] = action
        elif dt == "hook":
            self.decisions.hook = action
        elif dt == "music_timing":
            self.decisions.music_timing = action
        elif dt == "zoom":
            self.decisions.zoom[item.target_id] = action
            self._prev_zoomed = action != "NONE"
            if self._prev_zoomed:
                self._last_zoom_end = self._out_t
        else:  # emphasis
            if action != "NONE":
                self.decisions.emphasis.add(item.candidate_word_ids[int(action.split("_")[1])])

    def preview(self, item: Item, action: str) -> Decisions:
        """A copy of the decisions with `action` applied for `item` (what the edit looks like if it is taken)."""
        d = copy.deepcopy(self.decisions)
        dt = item.decision_type
        if dt == "zoom":
            d.zoom[item.target_id] = action
        elif dt == "rhythm":
            d.rhythm[item.target_id] = action
        elif dt == "pace":
            d.pace[item.target_id] = action
        elif dt == "hook":
            d.hook = action
        elif dt == "music_timing":
            d.music_timing = action
        elif dt == "emphasis":
            if action != "NONE":
                d.emphasis.add(item.candidate_word_ids[int(action.split("_")[1])])
        else:
            d.actions[item.target_id] = action
        return d

    def build_edl(self, face_samples=None, options: BuildOptions | None = None) -> EDL:
        return build_edl_with_lines(self.uf, self.decisions, face_samples, options)[0]


def make_suggestion(item: Item, choice: Choice, policy: Policy, video_id: str, source: str = "policy") -> Suggestion:
    return Suggestion(
        video_id=video_id, target_id=item.target_id, decision_type=item.decision_type,  # type: ignore[arg-type]
        proposed_action=choice.action, candidates=list(item.arms) if item.decision_type in ("emphasis", "hook") else [],
        context_raw={k: float(v) for k, v in item.ctx.items()}, propensity=choice.propensity,
        confidence=choice.confidence, explored=choice.explored, source=source,  # type: ignore[arg-type]
        policy_name=policy.name, policy_version=policy.version, feature_version=FEATURE_VERSION,
    )


def greedy_choice(policy: Policy, decision_type: str, ctx: dict[str, float],
                  offsets: dict[str, float] | None = None) -> Choice:
    """The posterior-mean argmax with no exploration (autopilot, PLAN 8.8): propensity 1; the confidence is the
    probability Thompson sampling would have given that action, so the review queue still ranks it honestly."""
    scores = policy.mean_scores(decision_type, ctx)
    for arm, o in (offsets or {}).items():
        if arm in scores:
            scores[arm] += o
    best = max(scores, key=lambda a: scores[a])
    probs = policy.choose(decision_type, ctx, offsets).probs
    return Choice(best, 1.0, probs.get(best, 0.5), False, probs)


def make_planner(uf: UnitsFile, video_type: str, policy: Policy, target: float | None, *, greedy: bool = False,
                 offsets: dict[str, dict[str, float]] | None = None, seed: int = 0):
    """The episode planner (app/rl/plan.py) when the video has a target duration and the policy is a linear bandit
    whose posterior can be sampled; None otherwise (then every decision is the bandit's)."""
    if not target or target <= 0 or not hasattr(policy, "_model"):
        return None
    from app.rl.plan import EpisodePlanner

    return EpisodePlanner(uf, video_type, policy, target, greedy=greedy, offsets=offsets, seed=seed)


PLANNER_TAG = "+psrl"  # appended to policy_name on suggestions the planner made


def planned_choice(planner, item: Item, sess: EditSession) -> tuple[Choice, str] | None:
    """(choice, posterior-mean plan's action) from the planner at the state the session is in, or None."""
    if planner is None or not planner.handles(item.decision_type, item.target_id):
        return None
    a, p, probs, mean_a = planner.choose(item.decision_type, item.target_id, sess._out_t, sess._since_cut)
    return Choice(a, p, p, a != mean_a, probs), mean_a


def suggest(uf: UnitsFile, word_feats: dict[int, dict[str, float]] | None, policy: Policy, *,
            video_type: str | None = None, offsets: dict[str, dict[str, float]] | None = None,
            source: str = "policy", greedy: bool = False, target: float | None = None,
            seed: int = 0, rhythm=None, hook=None) -> SuggestionSet:
    """Run `policy` over the whole video (every proposal is applied), returning suggestions + decisions.
    `greedy` = no exploration: the posterior mean decides (autopilot). With a `target` duration the phrase and
    pause decisions come from the episode planner (posterior sampling over the whole edit, app/rl/plan.py).
    With `rhythm` and `hook` agents the product's decision set is used (see EditSession.agents)."""
    agents = rhythm is not None
    sess = EditSession(uf, word_feats, video_type, agents=agents)
    if agents:
        rhythm.begin_video()
    planner = make_planner(uf, sess.video_type, policy, target, greedy=greedy, offsets=offsets, seed=seed)
    sugs, items, best = [], [], []
    for item in sess.items():
        off = (offsets or {}).get(item.decision_type)
        if item.decision_type in DEFAULT_OFFSETS:
            off = {**DEFAULT_OFFSETS[item.decision_type], **(off or {})}
        planned = planned_choice(planner, item, sess)
        agent_name = None
        if planned is not None:
            choice, mean_a = planned
            best.append(mean_a)
        elif item.decision_type == "rhythm":
            choice, agent_name = rhythm.choose(item.ctx), rhythm.name
            best.append(rhythm.mean_action(item.ctx))
        elif item.decision_type == "hook":
            choice, mean_a = hook.choose(item.ctx)
            agent_name = hook.name
            best.append(mean_a)
        else:
            choice = (greedy_choice(policy, item.decision_type, item.ctx, off) if greedy
                      else policy.choose(item.decision_type, item.ctx, off))
            scores = policy.mean_scores(item.decision_type, item.ctx)
            for arm, o in (off or {}).items():
                if arm in scores:
                    scores[arm] += o
            best.append(max(scores, key=lambda a: scores[a]))
        sug = make_suggestion(item, choice, policy, uf.video_id, source)
        if planned is not None:
            sug.policy_name += PLANNER_TAG
        if agent_name:
            sug.policy_name = agent_name
        sugs.append(sug)
        items.append(item)
        sess.commit(item, choice.action)
    return SuggestionSet(sugs, items, sess.decisions, sess.skipped_fillers, best)
