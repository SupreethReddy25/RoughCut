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
# frontend typecheck / unit+component tests / build (the built app is also served by the backend at /)
cd frontend; npm run typecheck; npm test; npm run build
# run a throw-away backend on another port with its own data dir (does not touch data/)
$env:ROUGHCUT_DATA_DIR="C:\tmp\rc"; cd backend; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8001
# regenerate the browser/Python parity fixtures (checked by backend/tests/test_frontend_fixtures.py)
backend\.venv\Scripts\python.exe backend\scripts\make_frontend_fixtures.py
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

## Product features (Phase 5)
- Background music: bundled loops in `backend/assets/music` are synthesised (CC0), rebuild with `backend\.venv\Scripts\python.exe backend\scripts\make_music.py`; uploads live in `data/music/u<user>/`. The mix (loop, 1 s fades, sidechain ducking, loudnorm on the mix) is in `app/render/graph.py`; the browser approximates it in `MusicPreview`.
- A new project starts from the user's remembered settings (`User.defaults_json` copied into `Video.settings_json["defaults"]`; they are a starting point, not events) and from a suggested title (overlay id `hook`, deleted/edited with normal text-overlay events).
- Batch / autopilot videos (`Video.autopilot`): greedy suggestions (`source = "autopilot"`), badge = "ready" when sum(1 - confidence) < 3 and trust >= 0.85 (`svc/projects.py`). One-click export skips finalize on purpose: nothing was reviewed, so nothing is learned.
- One nudge click moves at most a quarter of each decision type (`svc/rescore.py`).
