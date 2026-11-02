"""What did the model learn? For two contrasting creators: the posterior-mean advantage of one arm over another for hand-built
contexts, averaged over seeds. No prior, so everything shown comes from the creator's feedback.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\learned_weights.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from app.policy import context
from sim.experiments.common import make_learner, videos_for
from sim.personas import make_persona
from sim.session import run_user

OUT = Path(__file__).resolve().parents[1] / "data" / "learned.json"


def filler(tok):
    return {**{f"tok_{t}": float(t == tok) for t in context.fl.TOKENS}, "pos_mid": 1.0, "token_duration": 0.2, "gap_before": .1, "gap_after": .1,
            "word_conf": .9, "type_vlog": 1.0, "bias": 1.0}


def pause(g, end=1.0):
    return {"gap_seconds": g, "after_sentence_end": end, "before_new_sentence": end, "gap_rms": .05, "gap_voiced_ratio": 0.0, "position": .5,
            "hist_output_ratio": .8, "type_vlog": 1.0, "bias": 1.0}


def phrase(take):
    return {"duration": 3.0, "speech_rate": 3.0, "avg_word_conf": .9, "loudness_z": 0.0, "retake_score": 1.0 if take else 0.1, "is_earlier_take": float(take),
            "is_incomplete": float(take), "position": .5, "face_present": 1.0, "topic_centrality": .5, "hist_output_ratio": .8, "hist_since_last_cut": 3.0,
            "type_vlog": 1.0, "bias": 1.0}


CASES = {"filler um, REMOVE - KEEP": ("filler", filler("um"), "REMOVE", "KEEP"), "filler like, REMOVE - KEEP": ("filler", filler("like"), "REMOVE", "KEEP"),
         "pause 0.6 s, REMOVE - KEEP": ("pause", pause(.6), "REMOVE", "KEEP"), "pause 1.5 s, REMOVE - KEEP": ("pause", pause(1.5), "REMOVE", "KEEP"),
         "earlier take, CUT - KEEP": ("phrase", phrase(True), "CUT", "KEEP"), "ordinary phrase, CUT - KEEP": ("phrase", phrase(False), "CUT", "KEEP")}

if __name__ == "__main__":
    out = {}
    for name in ("snappy", "relaxed", "storyteller"):
        vals = {k: [] for k in CASES}
        for seed in range(1, 7):
            learner = make_learner("lints", None, seed, 0.995)
            run_user(make_persona(name), learner, videos_for(seed, 10), seed)
            for k, (dt, ctx, a, b) in CASES.items():
                s = learner.policy.mean_scores(dt, ctx)
                vals[k].append(s[a] - s[b])
        out[name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in vals.items()}
        print(name, {k: round(v["mean"], 2) for k, v in out[name].items()}, flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
