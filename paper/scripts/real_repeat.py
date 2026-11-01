"""How repeatable is the analysis? The same clips through the whole pipeline in fresh processes: paper/data/real_repeat.json.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\real_repeat.py [runs]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "data" / "real_repeat.json"
CLIPS = {"tarun": ROOT / "RAW Footage/Tarun.mp4", "aroll": ROOT / "RAW Footage/Raw Footage for Reel editing A-Roll.mp4"}
WORK = Path("C:/tmp/rc_paper/repeat")


def once(name: str) -> dict:
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

    uf = run_pipeline([CLIPS[name]], video_id=name, video_type="tutorial", out_dir=WORK / name, progress=lambda *_: None)
    wf = {int(k): v for k, v in json.loads((WORK / name / "word_features.json").read_text(encoding="utf-8")).items()}
    generic, protos = load_shipped_priors()
    mix = StyleMixture(protos, generic, sigma=STYLE_SIGMA, prior_sigma=STYLE_PRIOR_SIGMA)
    pol = LinTS(mix.mean_prior(), lam=PRODUCT_LAMBDA, v=PRODUCT_V, eps_floor=PRODUCT_EPS, seed=1, lam_by_type=PRODUCT_LAMBDA_BY_TYPE,
                eps_by_type=PRODUCT_EPS_BY_TYPE)
    ss = suggest(uf, wf, pol, greedy=True, rhythm=RhythmAgent(seed=1, greedy=True), hook=HookBandit(seed=1, greedy=True))
    edl = build_edl_with_lines(uf, ss.decisions)[0]
    text = " ".join(w.text_original for w in uf.words)
    return {"md5": hashlib.md5(text.encode()).hexdigest()[:10], "words": len(uf.words), "first_cut_s": edl.duration, "text": text,
            "first_cut": " ".join(x.text for ln in edl.captions.lines for x in ln.words),
            "crew": [u.text for u in uf.units if u.kind == "speech" and u.features.get("crew_score", 0) >= 0.5]}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        print("RESULT" + json.dumps(once(sys.argv[2])), flush=True)
        sys.exit(0)
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out: dict = {n: [] for n in CLIPS}
    for i in range(runs):
        for n in CLIPS:
            p = subprocess.run([sys.executable, __file__, "--one", n], capture_output=True, text=True, encoding="utf-8", errors="replace")
            line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT")), None)
            if line:
                out[n].append(json.loads(line[6:]))
                print(n, i, out[n][-1]["md5"], out[n][-1]["first_cut_s"], len(out[n][-1]["crew"]), flush=True)
            else:
                print(n, i, "FAILED", p.stderr[-400:], flush=True)
            OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
