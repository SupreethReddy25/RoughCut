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
# regenerate placeholder test clips into data/raw
backend\.venv\Scripts\python.exe scripts\make_test_clips.py
```
