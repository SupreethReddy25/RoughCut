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
# simulator experiments -> reports/figures/ (E1: learning curves, E2: style prior + decisions to identify the style, E3: cross-persona, E4: drift, E5: target length, bandit vs MDP planner, E8: review budget, E9: off-policy evaluation, E10: creators outside the model family, E11: rhythm, RLSVI vs bandit)
cd backend; .\.venv\Scripts\python.exe -m sim.experiments.e1; .\.venv\Scripts\python.exe -m sim.experiments.e3; .\.venv\Scripts\python.exe -m sim.experiments.e4
cd backend; .\.venv\Scripts\python.exe -m sim.experiments.e5; .\.venv\Scripts\python.exe -m sim.experiments.e9
# off-policy evaluation on a user's REAL logs (IPS / SNIPS / DR / SNDR with bootstrap CIs)
cd backend; .\.venv\Scripts\python.exe -m app.ope.run --user 1
# regenerate placeholder test clips into data/raw (clip_04_messy = English stress test: fillers, restarts, best take)
backend\.venv\Scripts\python.exe scripts\make_test_clips.py [--only clip_04_messy]
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

## Style prior and profile (Phase 6)
- `app/policy/style.py`: the style posterior (sigma 0.7, the sentence "Your style so far: ..." and "identified" above 0.8) and the prior mean it feeds the bandit (tempered with `prior_sigma` = 3 in the product so one click cannot swap the whole prior). `app/policy/preferences.py` turns the learned weights into profile sentences, `app/svc/stats.py` computes the editing statistics.
- `sim/review.py` + `sim/experiments/e8.py`: budgeted review sessions (uncertainty queue vs random vs timeline).

## Analysis quality (English)
- Transcription: faster-whisper `small.en` (`ROUGHCUT_WHISPER_MODEL` in .env), beam 10, no conditioning on previous text, words with no voice under them dropped AFTER alignment (`transcribe.drop_hallucinations`; Whisper's own timing can be seconds off). English only: other languages give a made-up transcript.
- Word timing: wav2vec2 CTC forced alignment (`app/pipeline/align.py`, `facebook/wav2vec2-base-960h`, `ROUGHCUT_ALIGN=0` to skip). Whisper's own timestamps hide pauses inside stretched words.
- Retakes (`features.py`): `false_start_matches` (function-word restarts matched against the speech that follows, across phrase boundaries), `retake_scores` (content-token retakes), `select_takes` (best take: a clean complete earlier delivery beats a worse repeat). `is_earlier_take` = "a better take of this line exists". `segment._split_restarts` splits restarts inside a phrase.
- Disfluency recovery (`align.recover_disfluencies`): wav2vec2's greedy reading restores stutters and hesitations Whisper tidied away. No-gap fillers are cut only where the audio dips at both edges (`fillers.snap_to_valleys`, `is_valley`).
- Fillers (`fillers.py`): stutters ("I I") use the `uh` token; hedges ("kind of", "sort of") and sentence openers ("Well,") use `other`.

## RL layers beyond the bandit (Phase 7)
- `app/ope/`: off-policy evaluation from logged propensities (DM, IPS, SNIPS, DR, SNDR; cross-fitted ridge reward model; cluster bootstrap CIs). Validated against simulator ground truth in E9.
- `app/rl/plan.py`: target-length MDP over the phrase/pause decisions, reward = the bandit's learned model with history features frozen at the bandit's own trajectory (letting them follow the state makes the solver cut everything: model exploitation). Solved exactly by backward induction; posterior sampling (PSRL) per video, closed-loop lookups, propensities from posterior draws. The target is a limit (`BETA` over, weak `BETA_UNDER` under), planned length is calibrated against the real EDL, and phrases of one sentence share their fate. Used when a video has `target_duration`; planned rows carry `policy_name` ending in `+psrl` and are not re-scored one by one.
- Product exploration (`app/appctx.py`): `PRODUCT_LAMBDA_BY_TYPE` (stronger prior on keep/cut) and `PRODUCT_EPS_BY_TYPE` (no random keep/cut flips). Experiments keep the plain defaults unless they say otherwise.
- Startup loads the analysis models in the background (`ROUGHCUT_PRELOAD=0` to skip). Non-English speech gets a warning banner (`svc/projects.language_warning`).
- `app/rl/rhythm.py`: visual rhythm agent (NONE, Z115, Z130, TEXT_POP, SFX per kept phrase; state = time since the last beat, its kind, the streak, the phrase), linear Q learned with RLSVI, the export rating is a delayed reward. `app/rl/hook.py`: opening title chosen by a Thompson bandit over candidate-line features; the footage is not reordered. Both are rebuilt from the log (`context.AGENT_TYPES` skip the bandit). `EditSession(agents=True)` is the product's decision set; simulator experiments keep the old one (zoom, no pace or hook).
- Phrase speed (`pace`, `Clip.speed`, handled by both `Remap`s) and music timing (`music_timing`, `MusicSpec.start_s`) are LinTS decisions with `session.DEFAULT_OFFSETS` so an untrained choice stays plain. `svc/bundle.NEEDS_EYE_CAP` limits how many cosmetic decisions ask for attention.
- `svc/progress.py`: analysing-screen stages and ETA, rates learned per machine in `data/eta.json`.
- `docs/DEMO.md` is the presentation runbook; `start_demo.ps1` starts the app.
