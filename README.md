# RoughCut

A video editor for talking-head Reels that makes the first cut by itself and learns each creator's editing style from their corrections, using reinforcement learning.

Upload a raw clip and RoughCut removes pauses, fillers, retakes and lines said to the crew, then asks only about the decisions it's unsure of. Each answer trains that creator's own agent, so the next video needs fewer corrections.

**Team:** M. Rushi Varun Reddy, P. Supreeth Reddy, B. Vijaysimha Reddy

## Run it

Needs Windows, Python 3.10 and Node, with ffmpeg in `tools/ffmpeg`. From the repo root in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\start_demo.ps1 -DataDir C:\tmp\rc_demo
```

The app opens at http://127.0.0.1:8000. Allow about a minute for the models to load. The first run on a new PC needs internet to download them.

First-time setup:

```powershell
cd backend; python -m venv .venv; .\.venv\Scripts\pip install -e ".[dev]"
cd ..\frontend; npm install; npm run build
```

## What it does

- **Analysis:** speech-to-text (Whisper), exact word timing (wav2vec2 forced alignment), pauses, fillers, stutters, retakes with best-take selection, crew chatter, face tracking and topic scoring. English only, up to 5 minutes per project.
- **Editing:** transcript editing, review queue with explanations, undo and redo, side-by-side compare, target length, captions, background music, reframing, batch mode.
- **Export:** 1080×1920 MP4 with captions and normalised loudness, plus an SRT file.

## The learning

| Part | Decides | Method |
|---|---|---|
| Main bandit | Keep/cut, pauses, fillers, highlights, phrase speed, music timing | Linear Thompson sampling, per creator |
| Style prior | Starting point for a new creator | Posterior over style prototypes |
| Length planner | Keep/cut and pauses for a target length | Finite-horizon MDP, backward induction, posterior sampling |
| Rhythm agent | Zoom, text pop-up or sound effect at each phrase | RLSVI, with the export rating as a delayed reward |
| Hook bandit | The opening title | Thompson sampling over line features |
| Off-policy evaluation | Value of another policy, from logs | IPS, SNIPS, DM, DR, SNDR |

## Results

These come from simulated creators. No study with real users has been run.

| Question | Result |
|---|---|
| Is it personal? | Own policy 0.936, another creator's 0.468 |
| Does the review queue help? | At 30% review: 0.869, random 0.784, timeline order 0.633 |
| Does planning hit a target length? | Length error 40% without it, 4.6% with it |
| Does RLSVI beat a bandit on rhythm? | No: 0.901 vs 0.919, with a fixed rule at 0.471 |

Full results and limits are in [docs/DEMO.md](docs/DEMO.md).

## Project layout

| Path | Contents |
|---|---|
| `backend/app/pipeline` | Video analysis |
| `backend/app/policy`, `backend/app/rl` | Bandits, planner, rhythm and hook agents |
| `backend/app/ope` | Off-policy evaluation |
| `backend/app/svc`, `backend/app/api` | Services and REST API |
| `backend/app/edl`, `backend/app/render` | Edit lists and ffmpeg rendering |
| `backend/sim` | Simulator and experiments E1–E11 |
| `frontend/src` | React and TypeScript editor |
| `paper/` | Paper, its data and the scripts that regenerate them |
| `docs/` | Presentation runbook, system diagram, archived notes |

## Tests and experiments

```powershell
backend\.venv\Scripts\python.exe -m pytest backend
cd frontend; npm run typecheck; npm test
cd ..; powershell -ExecutionPolicy Bypass -File paper\scripts\run_all.ps1
```

The last command regenerates every experiment, figure and number used in the paper. See [CLAUDE.md](CLAUDE.md) for all development commands and [PLAN.md](PLAN.md) for the full design.
