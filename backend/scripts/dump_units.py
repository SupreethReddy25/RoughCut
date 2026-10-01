"""Readable table of units.json: kind, start, end, text, key features.

    python backend/scripts/dump_units.py data/processed/clip_01_main            (folder or units.json)
    python backend/scripts/dump_units.py clip_01_main --words                    (id under data/processed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

KEYS = {
    "speech": ["duration", "speech_rate", "loudness_z", "filler_ratio", "retake_score", "is_earlier_take",
               "is_incomplete", "face_present", "semantic_redundancy", "topic_centrality"],
    "pause": ["gap_seconds", "gap_rms", "gap_voiced_ratio", "after_sentence_end", "prev_has_filler",
              "is_take_boundary"],
    "filler": ["token_duration", "gap_before", "gap_after", "likely_verb_like", "clean_gap", "word_conf"],
}
SHORT = {"speech_rate": "rate", "loudness_z": "loud_z", "filler_ratio": "fill", "retake_score": "retake",
         "is_earlier_take": "earlier", "is_incomplete": "incompl", "face_present": "face",
         "semantic_redundancy": "redund", "topic_centrality": "central", "gap_seconds": "gap",
         "gap_voiced_ratio": "voiced", "after_sentence_end": "after_sent", "prev_has_filler": "prev_fill",
         "is_take_boundary": "TAKE_B", "token_duration": "tok_dur", "likely_verb_like": "verb_like",
         "clean_gap": "clean", "word_conf": "conf", "duration": "dur", "gap_before": "gap_bef",
         "gap_after": "gap_aft", "gap_rms": "rms"}


def resolve(arg: str) -> Path:
    p = Path(arg)
    for cand in (p, REPO / "data" / "processed" / arg):
        if cand.is_dir():
            cand = cand / "units.json"
        if cand.is_file():
            return cand
    raise SystemExit(f"cannot find units.json for {arg!r}")


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--width", type=int, default=58, help="max text width")
    ap.add_argument("--words", action="store_true", help="also list words (id, times, conf, filler flag)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = resolve(a.target)
    d = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for u in d["units"]:
        counts[u["kind"]] = counts.get(u["kind"], 0) + 1
    print(f"{d['video_id']}  duration={d['duration']:.2f}s  type={d['video_type']}  "
          f"units={counts}  words={len(d['words'])}  source_map={d['source_map']}")
    print(f"{'id':>3} {'kind':<6} {'start':>7} {'end':>7}  {'text':<{a.width}}  key features")
    for u in d["units"]:
        ind = "  " if u["kind"] == "filler" else ""
        text = (ind + (u["text"] or "")).replace("\n", " ")
        if len(text) > a.width:
            text = text[: a.width - 1] + "…"
        f = u["features"]
        keys = KEYS[u["kind"]]
        extra = ""
        if u["kind"] == "filler":
            tok = next((k[4:] for k in f if k.startswith("tok_") and f[k] == 1.0), "?")
            pos = next((k[4:] for k in ("pos_start", "pos_mid", "pos_end") if f.get(k) == 1.0), "?")
            extra = f"[{tok}/{pos}] "
        feat = " ".join(f"{SHORT.get(k, k)}={fmt(f[k])}" for k in keys if k in f)
        print(f"{u['id']:>3} {u['kind']:<6} {u['start']:>7.2f} {u['end']:>7.2f}  {text:<{a.width}}  {extra}{feat}")
    if a.words:
        print("\nwords:")
        for w in d["words"]:
            flag = " FILLER" if w["is_filler"] else ""
            print(f"{w['id']:>4} u{w['unit_id']:<3} {w['start']:>7.2f} {w['end']:>7.2f} conf={w['conf']:.2f} "
                  f"{w['text_original']}{flag}")


if __name__ == "__main__":
    main()
