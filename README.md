# RoughCut

Human-in-the-loop RL for personalized short-form video editing: upload a talking-head clip, get a first cut, correct it, and the per-creator agent learns your style. English speech only.

**Run it:** `powershell -ExecutionPolicy Bypass -File start_demo.ps1` (opens http://127.0.0.1:8000).

- `docs/DEMO.md`: presentation runbook, results and known limits
- `PLAN.md`: full design
- `CLAUDE.md`: dev commands and module map

## Test clips are placeholders
`scripts/make_test_clips.py` generates synthetic clips (Windows TTS voice over a plain background) into `data/raw/`, including `clip_04_messy` (English stress test). Real face-to-camera clips are still needed for the dataset and the user studies.
