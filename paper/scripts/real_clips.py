"""Real-audio check of the analysis and the first cut a brand-new creator gets, against reference edits.

The references were written by Claude from each clip's transcript and from decoding doubtful stretches on their own, not
by a person listening. They use the transcript's wording, so this measures which lines the first cut keeps, not how well
words are heard. The crew rules were written while looking at the A-Roll clip, so that clip is not a held-out test of them.

    set PYTHONPATH=backend & set ROUGHCUT_DATA_DIR=C:\\tmp\\rc_paper
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\real_clips.py
"""
from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from app.appctx import (
    PRODUCT_EPS,
    PRODUCT_EPS_BY_TYPE,
    PRODUCT_LAMBDA,
    PRODUCT_LAMBDA_BY_TYPE,
    PRODUCT_V,
    STYLE_PRIOR_SIGMA,
    STYLE_SIGMA,
)
from app.edl.build import build_edl_with_lines
from app.pipeline.run import run_pipeline
from app.policy.linear import LinTS
from app.policy.prior import load_shipped_priors
from app.policy.style import StyleMixture
from app.rl.hook import HookBandit
from app.rl.rhythm import RhythmAgent
from app.session import suggest
from app.state import chosen_hook_overlay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "data" / "real_clips.json"
WORK = Path(__import__("os").environ.get("ROUGHCUT_DATA_DIR", "C:/tmp/rc_paper")) / "paper_clips"
CLIPS = {
    "real": (str(ROOT / "RAW Footage/Tarun.mp4"),
             "hello everyone welcome to the video editing course and my name is taran and this is your first assignment "
             "and in this assignment you'll learn how to add text inside the video how to remove the gaps inside the "
             "video how to color grade and how to remove this green screen and so much more you'll also be learning "
             "the difference between green screen and blue screen so that is it this is your first video edit this and "
             "send it to your mentors and i'll see you in the next assignment bye"),
    "aroll": (str(ROOT / "RAW Footage/Raw Footage for Reel editing A-Roll.mp4"),
              "this is what it looks like to edit a 21,000 rupees real estate video client wanted a clean 30 to 45 seconds "
              "edit what i got 160 gb of footage downloads failed files got corrupt hadn't even opened premiere pro yet and "
              "i already needed therapy i started assembly and then realised that half of the files got corrupt during "
              "download spent around 45 minutes finding the perfect music track send it to klein for approval immediately "
              "because i refuse to change it later all the clips were handled and very shaky once all the chaos was under "
              "control i quickly completed the rough edit added speed ramps smooth transitions and a few effects and a few "
              "sfx then came color correction and grading to give the footage a premium cinematic look i even added a 3d "
              "track title as a bonus just to leave a but in the end i started doubting the entire edit i was genuinely "
              "unsure if the client would like it still without overthinking more i send the final version to client for "
              "approval and to my surprise the client left a 5-star review and also came back for more future projects if "
              "you want to see the full step-by-step breakdown comment the word it to you you can check out the complete "
              "video below"),
    "scripted": (str(ROOT / "data/raw/clip_04_messy.mp4"),
                 "today i am going to show you how i light my videos the first thing you need is a big soft light put "
                 "it slightly to the side of your face the second tip is to use a darker background i think this is "
                 "the most underrated trick lighting matters more than the camera you use that is it for today let me "
                 "know what you want to see next"),
}


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9',-]*[a-z0-9]", s.lower().replace("’", "'"))


def main() -> None:
    generic, protos = load_shipped_priors()
    out = {}
    for name, (path, reference) in CLIPS.items():
        log: list[str] = []
        t0 = time.perf_counter()
        uf = run_pipeline([path], video_id=name, video_type="tutorial", out_dir=WORK / name, progress=log.append)
        secs = time.perf_counter() - t0
        wf = {int(k): v for k, v in json.loads((WORK / name / "word_features.json").read_text(encoding="utf-8")).items()}
        mix = StyleMixture(protos, generic, sigma=STYLE_SIGMA, prior_sigma=STYLE_PRIOR_SIGMA)
        pol = LinTS(mix.mean_prior(), lam=PRODUCT_LAMBDA, v=PRODUCT_V, eps_floor=PRODUCT_EPS, seed=1, lam_by_type=PRODUCT_LAMBDA_BY_TYPE,
                    eps_by_type=PRODUCT_EPS_BY_TYPE)
        t1 = time.perf_counter()
        ss = suggest(uf, wf, pol, greedy=True, rhythm=RhythmAgent(seed=1, greedy=True), hook=HookBandit(seed=1, greedy=True))
        edl = build_edl_with_lines(uf, ss.decisions)[0]
        suggest_s = time.perf_counter() - t1
        got = toks(" ".join(x.text for ln in edl.captions.lines for x in ln.words))
        ref = toks(reference)
        ops = [(op, ref[a:b], got[c:d]) for op, a, b, c, d in SequenceMatcher(None, ref, got, autojunk=False).get_opcodes()
               if op != "equal"]
        edits = sum(max(len(r), len(g)) for _, r, g in ops)
        kinds = {k: sum(u.kind == k for u in uf.units) for k in ("speech", "pause", "filler")}
        out[name] = {"source_s": uf.duration, "first_cut_s": edl.duration, "clips": len(edl.clips), "units": kinds,
                     "words": len(uf.words), "analysis_s": secs, "suggest_s": suggest_s, "n_suggestions": len(ss.suggestions),
                     "ref_words": len(ref), "got_words": len(got), "word_edits": edits, "wer": edits / len(ref),
                     "differences": [{"op": o, "ref": r, "got": g} for o, r, g in ops],
                     "first_cut_text": " ".join(got), "clip_ranges": [[round(c.src_start, 2), round(c.src_end, 2)] for c in edl.clips],
                     "low_confidence_decisions": sum(s.confidence < 0.7 for s in ss.suggestions), "steps": log,
                     "crew_lines_cut": [u.text for u in uf.units if u.kind == "speech" and u.features.get("crew_score", 0) >= 0.5],
                     "fillers_found": [u.text for u in uf.units if u.kind == "filler"],
                     "fillers_no_clean_cut": [u.text for u in uf.units if u.kind == "filler" and u.features.get("clean_gap", 1) < 1],
                     "retake_phrases_cut": sum(u.kind == "speech" and u.features.get("is_earlier_take", 0) >= .5 and ss.decisions.action(u.id) == "CUT" for u in uf.units),
                     "beats": {a: sum(x == a for x in ss.decisions.rhythm.values()) for a in set(ss.decisions.rhythm.values())},
                     "suggestions_by_type": {k: sum(s.decision_type == k for s in ss.suggestions) for k in {s.decision_type for s in ss.suggestions}},
                     "hook": getattr(chosen_hook_overlay(uf, ss.decisions.hook), "text", None)}
        print(name, json.dumps({k: v for k, v in out[name].items() if k not in ("steps",)})[:1500], flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
