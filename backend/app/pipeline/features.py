"""Unit assembly + static context features (PLAN 6.1, 6.3).

Turns words + speech spans + pause spans + filler groups (+ audio energy, face track, embeddings) into a
`UnitsFile`. Only *static* features live here. History features (hist_output_ratio, hist_since_last_cut,
hist_since_last_zoom, hist_prev_zoomed) depend on decisions made so far, so session.py computes them at
suggestion time; the video-type one-hot and bias are likewise added when the policy builds its vector.

Unit ids run in time order: speech unit, then its filler units, then the pause that follows it.
Per-word D5 features (word_loudness_z, is_number, ...) don't fit schemas.Word, so they are returned
separately and stored as the word_features.json sidecar. position_in_line is computed later, once
caption lines exist.
"""
from __future__ import annotations

import math
import re

import numpy as np
from rapidfuzz import fuzz

from app.pipeline import fillers as fl
from app.pipeline.pauses import Energy, PauseSpan
from app.pipeline.segment import is_sentence_end
from app.pipeline.semantic import STOPWORDS, redundancy_centrality, tfidf_scores, word_tfidf
from app.pipeline.vision import FaceSample, face_stats
from app.pipeline import crew
from app.schemas import SCHEMA_VERSION, Unit, UnitsFile, Word

FEATURE_VERSION = 1

# function words a finished sentence can't end on (so a phrase ending here was cut off / abandoned)
DANGLING = frozenset(["the", "a", "an", "and", "but", "or", "to", "of", "for", "with", "in", "on", "at", "that", "because", "if", "my", "your", "our", "their", "this", "these", "those", "as", "than", "from", "by", "into"])
DISCOURSE_STARTERS = frozenset(["so", "but", "and", "now", "okay", "alright", "anyway", "first", "second", "third", "next", "finally", "also", "then", "however"])
NUMBER_WORDS = frozenset(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "percent"])
NEGATIONS = frozenset(["no", "never", "not", "none", "nothing", "nobody", "nowhere", "neither", "nor", "cannot"])
I_FORMS = frozenset(("i", "i'm", "i'll", "i've", "i'd"))
EMPHASIS_PUNCT = re.compile(r"[!?]")
RETAKE_WINDOW = 3
EARLIER_TAKE_THR = 0.85


def _f(x: float, nd: int = 4) -> float:
    x = float(x)
    return round(x, nd) if math.isfinite(x) else 0.0


def _z(vals: list[float]) -> list[float]:
    if not vals:
        return []
    a = np.asarray(vals, dtype=np.float64)
    sd = a.std()
    return [0.0 if sd < 1e-6 else float(v) for v in (a - a.mean()) / (sd if sd >= 1e-6 else 1.0)]


def _norm(text: str) -> str:
    return fl.norm(text)


# ------------------------------------------------------------------ retakes
def content_tokens(words: list[Word]) -> list[str]:
    """Non-filler, non-stopword normalised tokens of a phrase (what retake matching compares)."""
    out = []
    for w in words:
        if w.is_filler:
            continue
        t = re.sub(r"[^a-z0-9']", "", _norm(w.text_original))
        if t and t not in STOPWORDS:
            out.append(t)
    return out


def _coverage(a: list[str], b: list[str]) -> float:
    """Share of the tokens of `a` that (fuzzily) occur in `b`."""
    if not a:
        return 0.0
    return sum(any(fuzz.ratio(t, u) >= 85 for u in b) for t in a) / len(a)


def retake_scores(token_lists: list[list[str]], window: int = RETAKE_WINDOW,
                  earlier_thr: float = EARLIER_TAKE_THR) -> tuple[list[float], list[int]]:
    """retake_score[i] = best rapidfuzz token_set_ratio (0..1) of phrase i vs the next 1..3 phrases.

    token_set_ratio is 100 when phrase i's tokens are a subset of a later phrase's, which is exactly an
    abandoned start followed by a full retake. Phrases with < 3 content tokens are damped so "okay so" can't
    match everything, and the score is scaled down when fewer than 85% of phrase i's tokens recur in the later
    phrase (list items such as "how to remove the gaps" / "how to remove the green screen" share words but are
    not retakes). is_earlier_take[i] = 1 when some later phrase in the window scores >= earlier_thr AND has at
    least as many content tokens (the later one is the more complete delivery). Tiny fragments (< 2 content
    tokens) do not use up the window: stammered restarts sit between an abandoned line and its retake.
    """
    n = len(token_lists)
    strs = [" ".join(t) for t in token_lists]
    scores, earlier = [0.0] * n, [0] * n
    for i in range(n):
        if len(token_lists[i]) < 2:
            continue
        damp = min(1.0, len(token_lists[i]) / 3.0)
        seen = 0
        for j in range(i + 1, n):
            if len(token_lists[j]) < 2:
                continue
            seen += 1
            if seen > window:
                break
            cov = _coverage(token_lists[i], token_lists[j])
            r = fuzz.token_set_ratio(strs[i], strs[j]) / 100.0 * damp * (1.0 if cov >= 0.85 else cov)
            scores[i] = max(scores[i], r)
            if r >= earlier_thr and len(token_lists[j]) >= len(token_lists[i]):
                earlier[i] = 1
    return scores, earlier


FALSE_START_GAP = 12.0  # s from the end of an abandoned start to the start of its retake
FALSE_START_SHORT_GAP = 4.0  # ... for a fragment of <= 3 words, which only counts if the restart follows quickly
FALSE_START_THR = 0.8


def false_start_matches(word_lists: list[list[str]], starts: list[float], ends: list[float],
                        sentence_end: list[bool], content: list[list[str]]) -> list[tuple[float, int, int]]:
    """Per phrase i: (score 0..1, j, m): phrase i is said again starting at later phrase j, and the repeat is the
    m words of speech that follow from j (they may run across phrase boundaries: a breath inside the retake
    splits it into several phrases). j = m = -1 when there is no match.

    Works on ALL words (stopwords included), because false starts are mostly function words ("you'll also be,",
    "and the,") that content-token matching cannot see. Phrase i must have >= 2 words and reappear at the start
    of a later phrase within FALSE_START_GAP seconds (any number of fragments may sit in between), and the
    speech from there must be at least as long as phrase i (fragments of <= 3 words: at least 2 words
    longer). Fragments may also match inside the head of the repeat (a restart can drop the first word:
    "also be," -> "you'll also be learning").
    """
    n = len(word_lists)
    out = [(0.0, -1, -1)] * n
    for i in range(n):
        wi = word_lists[i]
        if len(wi) < 2:
            continue
        short = len(wi) <= 3
        si = " ".join(wi)
        for j in range(i + 1, n):
            gap = starts[j] - ends[i]
            if gap > (FALSE_START_SHORT_GAP if short else FALSE_START_GAP):
                break
            stream: list[str] = []
            for q in range(j, n):  # the speech that follows from phrase j on
                stream += word_lists[q]
                if len(stream) > len(wi) + 2:
                    break
            # the repeat must go on further: a fragment (even one Whisper ended with a full stop, "The second
            # tip.") is restarted and finished; the same short line said twice ("Bye now. Bye now.") is not
            if len(stream) < len(wi) + (2 if short else 0):
                continue
            best_r, best_k = -1.0, len(wi)
            for k in range(max(1, len(wi) - 1), min(len(stream), len(wi) + 2) + 1):
                r = fuzz.ratio(si, " ".join(stream[:k]))
                if r > best_r:
                    best_r, best_k = r, k
            if short:
                best_r = max(best_r, fuzz.partial_ratio(si, " ".join(stream[:len(wi) + 2])))
            r = best_r / 100.0
            head_content = [t for t in stream[:best_k + 2] if t not in STOPWORDS]
            if content[i] and _coverage(content[i], head_content) < 0.7:
                r = min(r, 0.5)  # the head matches but what was said differs: a new sentence, not a restart
            if r > out[i][0]:
                out[i] = (r, j, best_k)
    return out


def false_start_scores(word_lists: list[list[str]], starts: list[float], ends: list[float],
                       sentence_end: list[bool], content: list[list[str]]) -> list[float]:
    """Just the scores of `false_start_matches`."""
    return [s for s, _, _ in false_start_matches(word_lists, starts, ends, sentence_end, content)]


BEST_TAKE_MARGIN = 0.08  # an earlier take must be clearly better than the repeat to be kept instead


def take_quality(conf: list[float], n_filler: int, n_words: int, complete: bool) -> float:
    """How well a take was delivered: recogniser confidence (clear speech), few fillers, and a finished sentence."""
    if n_words == 0:
        return -1.0
    return float(np.mean(conf)) - 0.6 * n_filler / n_words + (0.15 if complete else 0.0)


def select_takes(sp_words: list[list[Word]], word_lists: list[list[str]], matches: list[tuple[float, int, int]],
                 sentence_end: list[bool], incomplete: list[bool]) -> dict[int, int]:
    """Best-take selection. Normally the repeat wins (people redo a line because the first try was worse), but
    when a COMPLETE earlier take is clearly better delivered than its repeat (fewer fillers / restarts, clearer
    speech), the earlier take is kept and the repeat is dropped.

    Returns {phrase index -> index of the phrase that replaces it} for the phrases to drop."""
    n = len(sp_words)
    drop: dict[int, int] = {}
    for i, (r, j, m) in enumerate(matches):
        if r < FALSE_START_THR:
            continue
        drop[i] = j
        if not sentence_end[i] or incomplete[i]:
            continue  # an unfinished earlier take never beats the repeat
        # the phrases that make up the repeat (the m words from j on) must be just this line again
        rep, cnt = [], 0
        for q in range(j, n):
            rep.append(q)
            cnt += len(word_lists[q])
            if cnt >= m:
                break
        if cnt > m + 2 or not sentence_end[rep[-1]] or any(q in drop for q in rep):
            continue
        rw = [w for q in rep for w in sp_words[q]]
        qi = take_quality([w.conf for w in sp_words[i]], sum(w.is_filler for w in sp_words[i]), len(sp_words[i]), True)
        qr = take_quality([w.conf for w in rw], sum(w.is_filler for w in rw), len(rw), not incomplete[rep[-1]])
        if qi > qr + BEST_TAKE_MARGIN:
            del drop[i]
            for q in rep:
                drop[q] = i
    return drop


def _abandoned(k: int, drop: dict[int, int], sp_words: list[list[Word]]) -> bool:
    """A phrase that stops on "the"/"and"/... was abandoned unless the next phrase simply carries on with the
    sentence after a hesitation ("how to remove the" ... "gaps inside the video")."""
    if k in drop or k + 1 >= len(sp_words):
        return True
    nxt = next((w for w in sp_words[k + 1] if not w.is_filler), sp_words[k + 1][0])
    t = nxt.text_original
    return (t[:1].isupper() and _norm(t) not in I_FORMS) or _norm(t) in DISCOURSE_STARTERS


# ------------------------------------------------------------------ small word predicates
def _is_number(text: str) -> bool:
    t = _norm(text)
    return bool(re.search(r"\d", t)) or t in NUMBER_WORDS or t.rstrip("s") in NUMBER_WORDS


def _is_negation(text: str) -> bool:
    t = _norm(text)
    return t in NEGATIONS or t.endswith("n't")


def _is_incomplete(words: list[Word]) -> bool:
    ws = [w for w in words if not w.is_filler] or words
    # An ellipsis alone proves nothing: the disfluency prompt makes Whisper end ordinary sentences with "...".
    last = ws[-1].text_original
    return last.endswith(("-", "—")) or _norm(last) in DANGLING


# ------------------------------------------------------------------ assembly
def build_units(words: list[Word], spans: list[tuple[int, int]], pauses: list[PauseSpan],
                groups: list[fl.FillerGroup], *, video_id: str, duration: float, video_type: str = "other",
                source_map: list[dict] | None = None, energy: Energy | None = None,
                face_samples: list[FaceSample] | None = None, face_sampled: float = 0,
                emb: np.ndarray | None = None) -> tuple[UnitsFile, dict[int, dict[str, float]]]:
    """Returns (UnitsFile, word_features keyed by word id). Mutates words (unit_id)."""
    n_sp = len(spans)
    face_samples = face_samples or []
    has_track = face_sampled > 0

    # -- per-word audio + text features (sidecar)
    w_db = [energy.level_db(w.start, w.end) if energy else 0.0 for w in words]
    w_z = _z(w_db) if energy else [0.0] * len(words)
    tf = tfidf_scores([[w.text_original for w in words[a:b]] for a, b in spans])
    word_feats: dict[int, dict[str, float]] = {}
    for i, w in enumerate(words):
        prev_ends = i == 0 or is_sentence_end(words[i - 1].text_original)
        t = w.text_original
        cap = t[:1].isupper() and not prev_ends and _norm(t) not in I_FORMS
        word_feats[w.id] = {
            "is_number": float(_is_number(t)),
            "is_capitalized_mid_sentence": float(cap),
            "word_len": float(len(re.sub(r"\W", "", t))),
            "tfidf_in_transcript": _f(word_tfidf(t, tf)),
            "word_loudness_z": _f(w_z[i]),
            "is_last_in_sentence": float(is_sentence_end(t) or i == len(words) - 1),
            "is_negation": float(_is_negation(t)),
            "is_stopword": float(_norm(t) in STOPWORDS),
        }

    # -- speech-unit level numbers needed by several features
    sp_words = [words[a:b] for a, b in spans]
    sp_start = [w[0].start for w in sp_words]
    sp_end = [w[-1].end for w in sp_words]
    u_db = [energy.level_db_words(sp_words[k]) if energy else 0.0 for k in range(n_sp)]
    u_z = _z(u_db) if energy else [0.0] * n_sp
    sp_content = [content_tokens(w) for w in sp_words]
    rscore, rearlier = retake_scores(sp_content)
    sp_lists = [[t for t in (re.sub(r"[^a-z0-9']", "", _norm(x.text_original)) for x in w if not x.is_filler) if t]
                for w in sp_words]
    sp_send = [is_sentence_end(w[-1].text_original) for w in sp_words]
    sp_incomplete = [_is_incomplete(w) for w in sp_words]
    matches = false_start_matches(sp_lists, sp_start, sp_end, sp_send, sp_content)
    drop = select_takes(sp_words, sp_lists, matches, sp_send, sp_incomplete)
    for k in range(n_sp):
        if k in drop:  # a better take of this line exists: is_earlier_take is "drop me for the better take"
            rscore[k] = max(rscore[k], matches[k][0], FALSE_START_THR)
            rearlier[k] = 1
        elif matches[k][0] >= FALSE_START_THR:  # the better take that was kept: not a retake to cut
            rscore[k], rearlier[k] = min(rscore[k], 0.5), 0
    if emb is not None and len(emb) == n_sp:
        red, cen = redundancy_centrality(emb)
    else:
        red, cen = np.zeros(n_sp), np.zeros(n_sp)

    groups_by_span: dict[int, list[fl.FillerGroup]] = {}
    for g in groups:
        groups_by_span.setdefault(g.span_idx, []).append(g)
    pause_after = {p.prev_span: p for p in pauses}
    n_pause = len(pauses)
    pause_rank = {id(p): r for r, p in enumerate(pauses)}

    units: list[Unit] = []
    sp_uid: dict[int, int] = {}

    def add(kind, start, end, text, feats, parent=None) -> int:
        uid = len(units)
        units.append(Unit(id=uid, idx=uid, kind=kind, parent_unit_id=parent, start=_f(start), end=_f(end),
                          text=text, features={k: _f(v) for k, v in feats.items()}))
        return uid

    for k, ws in enumerate(sp_words):
        start, end = sp_start[k], sp_end[k]
        dur = end - start
        nonfill = [w for w in ws if not w.is_filler]
        fs = face_stats(face_samples, start, end, has_track=has_track)
        feats = {
            "duration": dur,
            "speech_rate": len(ws) / max(dur, 0.1),
            "avg_word_conf": float(np.mean([w.conf for w in ws])),
            "loudness_z": u_z[k],
            "filler_ratio": sum(w.is_filler for w in ws) / len(ws),
            "starts_with_filler": float(ws[0].is_filler),
            "nonword_voiced_ratio": _nonword_voiced_ratio(energy, ws, start, end),
            "retake_score": rscore[k],
            "is_earlier_take": float(rearlier[k]),
            "is_incomplete": float(sp_incomplete[k] and _abandoned(k, drop, sp_words)),
            "position": k / (n_sp - 1) if n_sp > 1 else 0.0,
            "is_first": float(k == 0),
            "is_last": float(k == n_sp - 1),
            "face_present": fs.present,
            "face_size": fs.size,
            "face_cx": fs.cx,
            "face_cy": fs.cy,
            "semantic_redundancy": float(red[k]),
            "topic_centrality": float(cen[k]),
            "has_emphasis_punct": float(bool(EMPHASIS_PUNCT.search(" ".join(w.text_original for w in ws)))),
            "emb_row": float(k),
            "n_words": float(len(ws)),
            "n_content_words": float(len(nonfill)),
        }
        uid = add("speech", start, end, " ".join(w.text_display for w in ws), feats)
        sp_uid[k] = uid
        for w in ws:
            w.unit_id = uid
        for g in groups_by_span.get(k, []):
            gw = [words[i] for i in g.word_idxs]
            before, after, clean = fl.clean_gap(words, g, energy)
            ab, bb = g.word_idxs[0], g.word_idxs[-1]
            ffeats = {f"tok_{t}": float(t == g.token) for t in fl.TOKENS}
            ffeats.update({
                "pos_start": float(ab == spans[k][0]),
                "pos_end": float(bb == spans[k][1] - 1),
                "token_duration": gw[-1].end - gw[0].start,
                "gap_before": min(before, 1.0),
                "gap_after": min(after, 1.0),
                "word_conf": float(np.mean([w.conf for w in gw])),
                "likely_verb_like": float(g.likely_verb_like),
                "loudness_z": float(np.mean([w_z[i] for i in g.word_idxs])),
                "clean_gap": float(clean),
            })
            ffeats["pos_mid"] = float(not ffeats["pos_start"] and not ffeats["pos_end"])
            add("filler", gw[0].start, gw[-1].end, " ".join(w.text_display for w in gw), ffeats, parent=uid)
        p = pause_after.get(k)
        if p is not None:
            prev_w, next_w = ws, sp_words[p.next_span]
            gap = p.end - p.start
            nxt_first = next(((w.text_original) for w in next_w if not w.is_filler), next_w[0].text_original)
            nxt_norm = _norm(nxt_first)
            pf = {
                "gap_duration": math.log1p(gap),
                "gap_bin_lt0.3": float(gap < 0.3),
                "gap_bin_0.3_0.6": float(0.3 <= gap < 0.6),
                "gap_bin_0.6_1.0": float(0.6 <= gap < 1.0),
                "gap_bin_1.0_2.0": float(1.0 <= gap < 2.0),
                "gap_bin_gt2.0": float(gap >= 2.0),
                "after_sentence_end": float(is_sentence_end(prev_w[-1].text_original)),
                "before_new_sentence": float((nxt_first[:1].isupper() and nxt_norm != "i")
                                             or nxt_norm in DISCOURSE_STARTERS),
                "gap_rms": energy.rel_level(p.start, p.end) if energy else 0.0,
                "gap_voiced_ratio": float(energy.voiced_mask(p.start, p.end).mean())
                if energy and energy.voiced_mask(p.start, p.end).size else 0.0,
                "prev_has_filler": float(any(w.is_filler for w in prev_w[-2:])),
                "is_take_boundary": float(p.is_take_boundary),
                "position": pause_rank[id(p)] / (n_pause - 1) if n_pause > 1 else 0.0,
                "gap_seconds": gap,
            }
            add("pause", p.start, p.end, "", pf)

    crew.annotate(units, words)
    uf = UnitsFile(schema_version=SCHEMA_VERSION, feature_version=FEATURE_VERSION, video_id=video_id,
                   duration=_f(duration), video_type=video_type, source_map=source_map or [], units=units,
                   words=words)
    return uf, word_feats


def _nonword_voiced_ratio(energy: Energy | None, ws: list[Word], start: float, end: float) -> float:
    """Voiced audio inside the phrase that no transcribed word accounts for (dropped um/uh, breaths)."""
    if energy is None:
        return 0.0
    sl = energy.span(start, end)
    voiced = energy.voiced[sl]
    if voiced.size == 0:
        return 0.0
    covered = np.zeros(len(energy.db), dtype=bool)
    for w in ws:
        covered[energy.span(w.start - 0.03, w.end + 0.03)] = True
    return float((voiced & ~covered[sl]).sum() / voiced.size)
