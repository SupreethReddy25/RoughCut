# RoughCut

Personalized short-form (talking-head) video editor: per-creator RL agent suggests edits, creator corrects, agent learns. Goal: replace a simple editor (CapCut-style) for Reels.

## Rules
- **PLAN.md is the source of truth.** Implement only the phase you're asked for; stop when its acceptance criteria pass.
- **Never change `backend/app/schemas.py` silently — ask first.** It's a frozen contract; `frontend/src/types.ts` mirrors it (`tests/test_schemas.py` checks sync).
- **ffmpeg lives in `tools/ffmpeg`** (project-local, gitignored). `app/config.py` finds it: `FFMPEG_DIR` (.env) -> `tools/ffmpeg/bin` -> PATH.
- Windows machine, Python 3.10, Node. Backend venv: `backend/.venv`.

## Commands (run from repo root, PowerShell)
```
# tests
backend\.venv\Scripts\python.exe -m pytest backend
# backend dev server (http://127.0.0.1:8000, docs at /docs)
cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
# frontend dev server (http://localhost:5173, proxies /api to :8000)
cd frontend; npm run dev
# frontend typecheck / build
cd frontend; npm run typecheck; npm run build
# analysis -> data/processed/{id}/ ; rule-based EDL -> edl.json ; render -> out.mp4 + out.srt
backend\.venv\Scripts\python.exe -m app.pipeline.run data\raw\clip_01_main.mp4 --type vlog
backend\.venv\Scripts\python.exe -m app.edl.run clip_01_main
backend\.venv\Scripts\python.exe -m app.render.run data\processed\clip_01_main\edl.json --out reports\demo.mp4
# simulator experiments -> reports/figures/ (E1: learning curves, E3: cross-persona, E4: drift)
cd backend; .\.venv\Scripts\python.exe -m sim.experiments.e1; .\.venv\Scripts\python.exe -m sim.experiments.e3; .\.venv\Scripts\python.exe -m sim.experiments.e4
# regenerate placeholder test clips into data/raw
backend\.venv\Scripts\python.exe scripts\make_test_clips.py
```

## Backend API (Phase 3)
- App factory `app.main:create_app` (context = DB + jobs + per-user policies, built at startup); data under `data/` (db.sqlite, processed/, uploads/, exports/, policies/).
- Shipped priors live in `backend/assets/priors` (rebuild: `python backend/scripts/build_prototypes.py`). Product bandit settings (lambda, style sigma) are in `app/appctx.py`.
- The event log is the source of truth: edit state, EDL and policy are recomputed from it (`app/state.py`, `app/policy/log_replay.py`).
