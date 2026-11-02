"""Crew chatter: lines spoken to the people behind the camera rather than to the viewer.

"Most of it will be voiceover, okay?" or "This is for YouTube Shorts." are clean, complete English, so retake and filler
detection pass them. Each sentence gets a score from what gives such a line away:

    production talk   words about making the video (strong) or checking in with someone (weak)
    quiet delivery    clearly softer than the creator's delivery to camera
    isolation         a long gap on both sides: said between takes
    short remark      a short sentence that also talks about production

and lines addressing the viewer ("comment", "subscribe", "check out") count against it. No single weak sign is enough:
a tutorial can mention a voiceover. A score of THRESHOLD or more makes the first cut remove the line, and the creator
can bring it back like any other cut. English only; the weights come from a small number of real clips.
"""
from __future__ import annotations

import re

from app.schemas import Unit, Word

THRESHOLD = 0.5
GROUP_GAP = 1.5  # phrases closer than this with no sentence end between them are one sentence
ISOLATED_GAP = 3.0
QUIET_Z = -0.9
SHORT_WORDS = 7
STRONG, WEAK, QUIET, ISOLATED, SHORT, VIEWER = 0.4, 0.2, 0.3, 0.15, 0.15, -0.25
_END = (".", "?", "!", "…")

STRONG_PATTERNS = [re.compile(p) for p in (
    r"\bvoice ?over\b", r"\bb-? ?roll\b", r"\ba-? ?roll\b",
    r"\bthis (one |part )?is for (youtube|instagram|shorts|reels|tiktok|facebook)",
    r"\bone for (instagram|youtube|shorts|reels|tiktok)",
    r"\b(two|three|different|another) (different )?(lines|takes|versions)\b",
    r"\b(is|are) (it|this|we|you) (rolling|recording)\b", r"\bfrom the top\b", r"\bone more (take|time)\b",
    r"\b(let me|lemme) (do|try|start|redo|say) (that|it|this)( one)? (again|over)\b", r"\b(retake|re-?shoot)\b",
    r"\btake (one|two|three|four)\b", r"\bstart (over|again)\b", r"\bcut (it|that|there)\b",
    r"\bwas that (ok|okay|good|fine|alright)\b", r"\b(keep|use) (this|that) (one|take)\b",
)]
WEAK_PATTERNS = [re.compile(p) for p in (
    r"\bokay\?$", r"\bok\?$", r"\bright\?$", r"\bwe'll (create|shoot|add|do|use|edit|fix)\b", r"\bi'll do\b", r"\bsorry\b",
    r"\bhold on\b", r"\bwait\b", r"\bshould i\b", r"\bdo you want\b", r"\bcan you (hear|see)\b",
)]
VIEWER_PATTERNS = [re.compile(p) for p in (
    r"\bcomment\b", r"\bsubscribe\b", r"\bfollow\b", r"\blink\b", r"\bcheck (it |this )?out\b", r"\blet me know\b", r"\byou'll\b",
)]


def score_text(text: str) -> float:
    t = text.lower().replace("’", "'").strip()
    s = STRONG * sum(bool(p.search(t)) for p in STRONG_PATTERNS)
    s += WEAK * sum(bool(p.search(t)) for p in WEAK_PATTERNS)
    s += VIEWER * sum(bool(p.search(t)) for p in VIEWER_PATTERNS)
    return s


def annotate(units: list[Unit], words: list[Word]) -> None:
    """Set features["crew_score"] (0..1) on every speech unit, in place. The whole sentence shares one score."""
    speech = [u for u in units if u.kind == "speech"]
    groups: list[list[Unit]] = []
    for u in speech:
        if groups:
            prev = groups[-1][-1]
            if not prev.text.rstrip().endswith(_END) and u.start - prev.end < GROUP_GAP:
                groups[-1].append(u)
                continue
        groups.append([u])
    for k, g in enumerate(groups):
        text = " ".join(u.text for u in g)
        lex = score_text(text)
        s = lex
        spoken = sum(u.end - u.start for u in g) or 1.0
        z = sum(u.features.get("loudness_z", 0.0) * (u.end - u.start) for u in g) / spoken
        before = g[0].start - groups[k - 1][-1].end if k else ISOLATED_GAP
        after = groups[k + 1][0].start - g[-1].end if k + 1 < len(groups) else ISOLATED_GAP
        if lex > 0:  # the delivery only counts as evidence when the words already point that way
            if z <= QUIET_Z:
                s += QUIET
            if before >= ISOLATED_GAP and after >= ISOLATED_GAP:
                s += ISOLATED
            if len(text.split()) <= SHORT_WORDS:
                s += SHORT
        for u in g:
            u.features["crew_score"] = round(max(min(s, 1.0), 0.0), 3)
