# Evidence for the paper

All numbers come from the final code. Regenerate them with `powershell -ExecutionPolicy Bypass -File paper\scripts\run_all.ps1`. `main.tex` is built from `main_base.tex` by `scripts/build_tex.py`, which fills in these results. After editing `main.tex` by hand, do not run that script again, or the edits are lost.

## Where each result lives

| Topic | File in `paper/data/` |
|---|---|
| E1–E11 summaries and figures | `e/` (E11 raw runs in `e/e11_raw.json`) |
| Baselines, ablation, sensitivity, robustness | `main.json`, `ablation.json`, `sensitivity.json`, `robustness.json` |
| Paired statistics | `paired.json` |
| Exploration on held-out videos | `exploration.json` |
| Calibration of confidence | `calibration.json` |
| Learned weights | `learned.json` |
| Runtime and model size | `runtime.json`, `new_features.json` (`cost`) |
| Final bandit settings, E11 paired comparisons | `new_features.json` |
| Real clips | `real_clips.json`, `real_repeat.json` |
| Every number for the tables | `numbers.json` |

## Key results

Simulated creators unless stated. CI = 95% confidence interval half-width.

| Result | Value |
|---|---|
| E1 per-type comparison at video 5 | 48 ahead, 2 tied, 1 behind, 9 at ceiling |
| E2 acceptance, videos 1–3 | none 0.824, generic 0.831, style mixture 0.845; style identified in a median of 3.5 events |
| E3 personalisation | own 0.936, other creators 0.468 |
| E4 drift | 0.948 before, 0.605 at the switch, above 0.80 within 3 videos |
| E5 length error, acceptance | bandit 40.0%, 0.883; with planner 4.6%, 0.879 |
| E8 at 30% review | uncertainty 0.869, random 0.784, timeline 0.633 |
| E9 true / DR | learned 0.932 / 0.945, rule 0.749 / 0.739, generic 0.703 / 0.694 |
| E9 CI coverage (nominal 95%) | DR 91%, SNDR 86%, SNIPS 81%, IPS 72%, DM 48%; ESS share 15% |
| E10 last video | nonlinear 0.883 (rule 0.511, fixed arm 0.489); noisy 0.810 (0.633, 0.835); arbitrary 0.499 (0.505, 0.568) |
| E11, 4 types × 20 unseen seeds | RLSVI 0.901, bandit 0.919, rule 0.471; rating +0.04, +0.04, −0.42 |
| E11 RLSVI minus bandit | matches −0.018 (CI 0.019, p = 0.055); rating +0.003 (CI 0.071, p = 0.93) |
| Final vs earlier bandit settings, 120 fresh runs, video 6 | 0.903 vs 0.892 (+0.011, CI 0.005) |
| Drift with final settings, 6 videos after the switch | 0.829 vs 0.814 (+0.015, CI 0.013, n = 20) |
| Neural bandit, 60 runs, same seeds as the other baselines | blank: 0.897 at video 10, regret 225.3; pretrained: 0.896, regret 217.8; proposed: 0.910, regret 181.5 |
| Tuned deep agents on rhythm (`deep_numbers.json`) | DQN 0.855 blank, 0.865 warm; PPO 0.532 blank, 0.542 warm; ratings +0.04, +0.03, −0.25, −0.11 |
| Cost | RLSVI 145 parameters, 16 ms to prepare a video after 10 videos, 0.08 ms per decision; planner about 186 ms per video |

## Real clips

| Clip | Source → first cut | Clips | Retake phrases cut | Crew lines cut | Analysis |
|---|---|---|---|---|---|
| Tarun | 44.2 s → 22.1 s | 6 | 4 | 0 | 119 s |
| A-Roll | 238.5 s → 72.6 s | 24 | 6 | 5 | 270 s |
| Stress clip (synthetic voice) | 54.5 s → 19.2 s | 12 | 5 | 0 | 98 s |

Analysis times were measured while the test suite ran, so they overstate an idle machine. Three fresh full runs gave identical text and cuts on Tarun and A-Roll.

## What the paper must state

- **No human study.** All learning results use personas we wrote, and the priors were trained on them.
- **Real-clip references are not independent.** Claude wrote them from transcripts and from decoding doubtful stretches, not a person listening, and they use the transcript's wording. Zero word differences on Tarun and A-Roll shows the first cut keeps the expected lines; it is not a transcription accuracy or a general edit accuracy figure.
- **Very few real clips.** Two real clips and one synthetic voice clip. `v1.mp4` duplicates `Tarun.mp4`. The crew rules and the Whisper fix were developed on these clips, so they are not held-out tests.
- **The Whisper isolation is a workaround.** Inside a process that had prepared footage, identical audio gave a different transcript (305 vs 324 words on A-Roll). Running Whisper in a fresh process fixed it. The internal cause was not found.
- **RLSVI does not beat the bandit.** Report it as a finding, and note that these personas react to state the bandit also sees.
- **E10.** LinTS is below the best fixed arm on noisy and arbitrary creators.
- **Audio has not been reviewed by ear.**
- **Settings used in experiments.** E1–E10 use the experiments' default settings unless stated; the product uses stronger priors and less exploration.
