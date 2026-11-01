"""Off-policy evaluation on the REAL logs of one user (PLAN 7.4): which policy would have served this creator best?

    python -m app.ope.run --user 1 [--data-dir data]

Uses the user's logged LinTS suggestions (with propensities) and explicit verdicts, and evaluates: the user's
current learned policy (posterior mean), the rule baseline (a simple editor's fixed thresholds), the generic prior
(no personalisation), and every style prototype.
"""
from __future__ import annotations

import argparse
import sys

from sqlmodel import select

from app.appctx import AppContext
from app.convert import event_from_row, suggestion_from_row
from app.models import EventRow, SuggestionRow, Video
from app.ope.estimators import ESTIMATORS, evaluate, greedy_target
from app.ope.logs import logged_from
from app.policy.linear import GlobalPolicy
from app.policy.rule import RulePolicy


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="off-policy evaluation on a user's real logs")
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ctx = AppContext(a.data_dir, workers=1)
    try:
        with ctx.session() as s:
            vids = [v.id for v in s.exec(select(Video).where(Video.user_id == a.user)).all()]
            sugs = {r.id: suggestion_from_row(r) for r in s.exec(select(SuggestionRow).where(
                SuggestionRow.video_id.in_(vids))).all()} if vids else {}  # type: ignore[attr-defined]
            evs = [event_from_row(r) for r in s.exec(select(EventRow).where(EventRow.user_id == a.user)
                                                     .order_by(EventRow.id)).all()]
            learner = ctx.policies.get(s, a.user)
        data = logged_from(sugs, evs)
        print(f"user {a.user}: {len(vids)} videos, {len(sugs)} suggestions, {len(data)} with an explicit verdict")
        if not data:
            print("nothing to evaluate yet: review some suggestions in the editor first")
            return
        targets = {"learned policy (now)": greedy_target(learner.policy),
                   "rule (simple editor)": (lambda rule: lambda dt, c, arms: {
                       x: float(x == rule.choose(dt, c).action) for x in arms})(RulePolicy(0))}
        if ctx.shipped_priors is not None:
            generic, protos = ctx.shipped_priors
            targets["generic prior"] = greedy_target(GlobalPolicy(generic))
            for name, p in protos.items():
                targets[f"style: {name}"] = greedy_target(GlobalPolicy(p))
        print(f"{'target policy':24}" + "".join(f"{e:>19}" for e in ESTIMATORS) + "    ESS")
        for name, t in targets.items():
            r = evaluate(data, t, n_boot=a.boot)
            cells = "".join(f"{r.estimates[e]:.2f} ({r.ci[e][0]:.2f}-{r.ci[e][1]:.2f})".rjust(19) for e in ESTIMATORS)
            print(f"{name:24}{cells}  {r.ess:5.1f}")
        print("value = expected share of suggestions the creator keeps; [95% bootstrap CI over videos]")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
