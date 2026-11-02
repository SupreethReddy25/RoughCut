# RoughCut: status and plan for the next Sonnet sessions

Written 2026-10-01 after Phase 6 (`99e5517`). PLAN.md stays the source of truth; this file says what is done, what
is not, and how to hand the rest to Sonnet one phase at a time.

## 1. Status

| Phase | State | Evidence |
|---|---|---|
| 0 Scaffolding | done | `1fd8590` |
| 1 Pipeline + render core | done | `3342597`, `cc26dc0` |
| 2 Policies + simulator, E1/E3/E4 | done | `acb635e`, `reports/figures/e1_*`, `e3_*`, `e4_*` |
| 3 Backend API | done | `cd289ee` |
| 4 Editor UI core | done | `b6a2e51` |
| 5 Product completeness (all 6 items) | done | `d1226bd` |
| 6 Style prior, queue polish, profile, E2/E8 | done | `99e5517`, `reports/figures/e2_*`, `e8_*` |
| **7 OPE (E9)** | **not started** | `backend/app/ope/` is an empty package |
| **8 Deployment + studies** | **not started** | `Dockerfile` is a placeholder; no study/admin endpoints |
| 9 Could items | not started (only after Phase 8 passes) | `backend/app/rl/` empty |
| 10 Report + demo | not started | |

Tests at `99e5517`: backend 276 passed (incl. slow real renders), frontend 159 passed; ruff, typecheck, build clean.

### Gaps against PLAN that are NOT in a later phase's own list (fold them into the phase noted)

| Gap | PLAN ref | Do it in |
|---|---|---|
| Profile management: `POST /api/users/{id}/reset`, `GET .../export`, `POST .../import` | 16.2 | Phase 8 |
| `POST /api/study/sessions` (model `StudySession` exists, no route) and `GET /api/admin/logs.csv` | 16.2 | Phase 8 |
| `scripts/export_logs.py` (daily log export, ephemeral hosting) | 15, Phase 8 | Phase 8 |
| E7 ablations (feature groups, implicit weight, seen-tracking, manual-edit mapping on/off) — the simulator half | 10, 17 | Phase 7 (alongside E9; same `sim/experiments` machinery) |
| E12 analysis (autopilot "Ready to post" correct with ≤ 2 changes, corrections/clip) | 10 | Phase 8 |
| Download `fmt=png` (cover) and finalize's "PPO retrain, ≤ 2 comparisons" | 16.2 | Phase 9 (Could) |
| A dedicated `POST /api/autopilot` batch endpoint | 16.2 | **not needed**: batch is implemented as `POST /api/projects` with `autopilot=true` per file (limit 5 waiting). Keep as is unless the report needs the batch id. |

### Known, deliberately left as is (decide before the report; none blocks Phase 7/8)
- Trust counts implicit accepts (seen, untouched) as accepts; PLAN 8.8 says "explicit-acceptance rate". Read as "reviewed = seen or decided". Change in `svc/projects.trust_score` if the team reads it literally.
- Undoing a setting does not undo the remembered default for the *next* project.
- The style posterior (sigma 0.7, PLAN value) shows "100% <style>" after about one video.
- A batch video is "ready" a second before its auto-edit exists (card shows "Auto-editing…").
- The profile recomputes per-video statistics on every request (fine for tens of videos).
- `Timeline.layoutClips` duplicates offsets that `lib/remap.ts` has.

### Things only the humans can do (Sonnet cannot)
1. **Record real clips.** `data/raw/` still has the TTS placeholders from `scripts/make_test_clips.py`: no real face, so
   face tracking / reframing has never been tested on a person. PLAN 19.4 says to make one real Reel end to end
   yourselves *before* more features. Do this now, before Phase 8.
2. Pick the host (HF Spaces / Render / college server) and create the account.
3. Run the studies (E6, E11, E12): participants, raters, SUS forms, consent.
4. Look at the plots (E1–E4, E2, E8) and write the report.

## 2. How to run a Sonnet session (unchanged from PLAN 19)

Start every session with:

> Read PLAN.md, CLAUDE.md and SONNET_PLAN.md. Implement **Phase N only** as described in SONNET_PLAN.md section 3.
> Stop when its acceptance criteria pass and show me the test output. Never change `backend/app/schemas.py`
> silently — ask first. Don't start the next phase.

### Environment notes Sonnet keeps re-learning (put them in the prompt if it struggles)
- Windows, Git Bash + PowerShell. Bash heredocs with quotes inside sometimes fail to parse: write files with the
  Write tool or a small Python patch script, and make patch scripts **assert** that the text they replace exists
  (a silent `str.replace` miss cost a debugging round twice).
- Never run a bare `python -` / `python` that waits on stdin; always `</dev/null` or a script file.
- The user's own dev server runs on **port 8000 — never stop it.** For end-to-end checks start a throw-away backend on
  8001 with `ROUGHCUT_DATA_DIR` pointing at a temp folder, stop only that process (by port), delete the folder.
- The built-in browser pane may not draw (requestAnimationFrame stops when the pane is hidden): screenshots time out
  and the preview's playback loop freezes. Use `get_page_text`/JS for checks; to test playback, patch
  `window.requestAnimationFrame` to a `setTimeout` shim and navigate client-side (SPA link click) so components re-mount.
- Full backend suite takes ~6–8 min (`pytest backend`); run it in the background. Frontend: `npm test` (~15 s).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or whatever model runs the session).

## 3. Remaining phases, ready to hand over

### Phase 7: OPE + E9 (+ the simulator half of E7)

**Build**
1. `app/ope/estimators.py` — pure functions over logged rows `(context_raw, decision_type, logged_action,
   propensity, reward, arms)`:
   - `ips`, `snips`, `dr` (q̂ = ridge per (decision_type, arm) on the same `context.vectorize`, **cross-fitted** in
     k folds so a row is never scored by a model trained on it), each returning value + bootstrap 95% CI
     (resample rows; fixed seed).
   - Target policies are evaluated as action *distributions* π_e(a|x): deterministic greedy = one-hot; LinTS uses
     `choose(...).probs`. Clip importance weights (report the clip value) and report effective sample size.
2. Logged data from the product: `app/ope/run.py` builds rows from `EventRow` + `SuggestionRow`:
   - include explicit `accept` / `reject` / `override` only (PLAN 7.4); reward = +1 accept, −1 for the proposed arm
     on reject/override (same numbers as `policy/reward.py`, import them, don't copy).
   - **exclude** `source in ("nudge", "autopilot")`, rescore rows with propensity 1.0 made by `:rescore`
     (deterministic: say so in the output), manual-only events (`via == "manual"` with no panel decision), undone events.
   - CLI: `python -m app.ope.run --user <id> [--policies rule,global,eps,linucb,lints]` → table of estimates + CIs.
3. E9 in the simulator: `sim/experiments/e9.py`
   - Behaviour policy LinTS (propensities logged) over personas × seeds via `run_user(..., keep_log=True)`
     (it already records suggestions with propensities and events).
   - Truth for each target policy = its acceptance against the persona's ideal on the SAME logged contexts
     (the simulator knows the ideal). Rewards in the log must use the same scale as the truth (map accept=1 /
     corrected=0 or ±1 consistently; document the choice).
   - Plot estimation error (IPS, SNIPS, DR) per target policy with CIs, and the ranking table vs the true ranking.
   - Note honestly in the docstring that decisions are sequential (history features follow the logged actions)
     and OPE treats them as independent bandit rounds.
4. E7 simulator ablations `sim/experiments/e7.py`: LinTS acceptance curve with one thing switched off at a time —
   each feature group (zero its columns), implicit weight ∈ {0, 0.5, 1}, seen-tracking off (implicit for unseen too),
   manual-edit mapping off (manual corrections give no update). Δ acceptance at video 5 with CIs, one bar chart.

**Accept:** SNIPS and DR within ±0.05 of the simulator truth for the target policies (or an honest report of where
they are not, with the reason), E9 and E7 plots + summary JSON in `reports/figures/`, unit tests for the estimators
on a hand-made log where the answer is known (uniform logging → IPS = mean reward; DR with a perfect q̂ = truth),
`ope.run` tested on an API-made log (fake pipeline, see `tests/apihelpers.py`).

### Phase 8: Deployment + study tooling

**Build**
1. **Docker image** (multi-stage): node stage runs `npm ci && npm run build`; Python stage installs the backend,
   copies `frontend/dist`, installs ffmpeg **with libass** (Debian's ffmpeg has it; the app checks
   `subtitles`, `sidechaincompress`, `loudnorm`, `afftdn`, `ebur128` at startup — make the container fail fast if
   missing), pre-downloads faster-whisper `base.en` (int8) and `all-MiniLM-L6-v2` at build time so the first upload
   does not download models, bundles fonts/music/priors (already under `backend/assets`). Data dir via
   `ROUGHCUT_DATA_DIR`, one uvicorn worker (the job runner and locks are in-process). For HF Spaces: listen on 7860,
   non-root user, persistent `/data` if available.
2. **Limits** (PLAN Phase 8): ≤ 3 min total (exists: `MAX_TOTAL_SECONDS`), ≤ 500 MB (exists), one processing job at
   a time (ingest pool already has 1 worker — verify renders cannot starve it on a 2-CPU host).
3. **API gaps** from section 1: profile reset / export / import (export = the user's events + suggestions +
   settings as JSON; import must go through the same `Learner` replay so the policy equals a rebuild),
   `POST /api/study/sessions`, `GET /api/admin/logs.csv` behind an `ADMIN_TOKEN` env var (no token → 404).
   `scripts/export_logs.py` writes the same CSV + DB copy with a date stamp.
4. **Study tooling (minimal, inside the app):**
   - E11: a `/study` page to enter a session (participant, condition, clip, seconds, n_actions); RoughCut's own
     time-to-export comes from `Metric.seconds_active` + render time, logged automatically.
   - SUS questionnaire page (10 items, stored per participant).
   - Blind pairwise rating page for raters: two exported videos side by side in random order, "which would you
     post?" win / tie / loss, stored with rater id. Export everything as CSV.
   - Analysis scripts `backend/scripts/analyze_e6.py`, `analyze_e11.py`, `analyze_e12.py` that read the exported
     logs and produce the figures/tables (write and test them on synthetic logs before the study).
5. README: how to build and run the image, how to deploy, how to export logs.

**Accept (Sonnet's part):** `docker build` succeeds and the container serves the app on a clean machine
(upload a test clip → suggest → export works inside the container), health check reports all filters, the new
endpoints and pages are tested, analysis scripts run on synthetic logs. **Humans' part:** public URL works, E11 data
collected, logs exported (PLAN Phase 8 acceptance).

### Phase 9 (Could — only after Phase 8 passes), in this order
1. Cover image (PLAN 8.11): pick a frame in the preview, add the title, export a 1080×1920 PNG (`download?fmt=png`).
2. Hook D6 (PLAN 6.2): once per video, move one of the top-3 phrases to the start (action-feature LinTS).
3. PPO + pacing comparisons (E5, PLAN 7.6) — `rl/edit_env.py`, `train.py`, `pacing_bt.py`; finalize offers ≤ 2
   "which feels better?" comparisons. Big item; start only if the report needs the MDP angle.
4. Neural-Linear (E10, PLAN 7.5).
5. LLM scoring.

### Phase 10: Report + demo
Sonnet can help draft: figure captions from the `*_summary.json` files, the RL vs product evaluation split, the
3-minute demo script (PLAN Phase 10) and a seeded demo data set (a user with 5 finished videos so the queue is short
and the style is identified). The writing and the claims are the team's.

## 4. Suggested order from here
1. Humans: record 3–5 real face-to-camera clips (one with two takes), make one real Reel end to end, note anything
   that breaks (especially reframing on a real face). Fix those first (one Sonnet session).
2. Sonnet: Phase 7 (OPE + E9 + E7 simulator ablations).
3. Sonnet: Phase 8 build items 1–5; humans deploy and run E6/E11/E12.
4. Sonnet: E7/E9 on the real logs (rerun `ope.run` and e7 on exported logs), then Phase 9 items only if time remains.
5. Phase 10.
