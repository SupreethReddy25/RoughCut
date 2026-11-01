# RoughCut presentation runbook

All numbers below come from the final code (`reports/figures/*_summary.json`). The simulator results are about simulated
creators: no human study (E11/E12 in PLAN) has been run, so do not claim one.

## Setup (do it 15 minutes before)
1. Use a fresh data folder, so old test projects (including garbage Telugu transcripts) don't appear:
   `powershell -ExecutionPolicy Bypass -File start_demo.ps1 -DataDir C:\tmp\rc_demo`
2. Wait about a minute: the models load in the background so the first upload is fast. Upload one clip as a rehearsal.
3. Before the talk, edit and export 2-3 English videos yourself, so Profile shows real learning curves. It is empty until then.
4. Keep a backup: an exported .mp4 and the figures in `reports/figures`, in case of a machine problem.
5. The analysis needs the models already cached on this PC (they are). A different PC needs internet once.

## Demo flow (about 6 minutes)
| Step | Do | Say |
|---|---|---|
| 1 | Upload an English clip with retakes (`data/uploads/v_9/00_Tarun.mp4`, type Tutorial). Takes ~70-90 s; narrate while it runs, or have one ready. | Whisper transcript, wav2vec2 forced alignment, pauses, fillers, retakes. |
| 2 | Open it: 44 s became ~22 s. Press Play. | First cut made by the bandit, before any human input. |
| 3 | Show struck lines in the transcript. | The abandoned takes were found; the final take is kept. |
| 4 | Review panel: press `J`, read "Why?", press `A` / `R`. | Uncertainty queue; explanation = feature contributions of a linear model. Every click is logged with its propensity. |
| 5 | `Ctrl+Z`, then delete a word in the transcript. | The event log is the source of truth; undo recomputes the policy. |
| 6 | Export... then play the file. | 1080x1920, captions, -14 LUFS. |
| 7 | Upload `data/raw/clip_04_messy.mp4` with target 20. Show the chip. | The MDP planner: the length holds, no half sentences. |
| 8 | Profile, then the figures. | What it learned, and the evidence. |

## Results to quote
| Experiment | Result |
|---|---|
| E1 learning | LinTS beats rule and global baselines overall by video 5 for every persona. Per decision type: 49 ahead, 1 tied, 1 behind, 9 at ceiling. |
| E2 cold start | Acceptance on videos 1-3: no prior 0.819, generic 0.831, style mixture 0.847. Style identified in a median of 3.5 events. |
| E3 personalisation | Own policy 0.937 vs another creator's 0.466. |
| E4 drift | After a style switch, recovers to 0.80 within 3 videos. |
| E5 planning | Held-out length error, measured on the real edit: bandit 40%, bandit + MDP planner 4%. Acceptance 0.888 vs 0.877: fitting the length costs about 1 point. |
| E8 review queue | At 30% review budget, posted quality 0.859 (uncertainty queue) vs 0.779 (random) vs 0.618 (timeline). No advantage at 100%, as expected. |
| E9 off-policy | True acceptance of learned LinTS 0.932 (SNDR estimate 0.944), rule 0.749 (0.739), generic 0.703 (0.697). DR's 95% interval covers the truth 89% of the time. |
| E10 wrong model | Creators outside the model family and the prior's training data, keep/cut + pause + filler only, last video. Nonlinear rules: LinTS 0.886, rule 0.509, best fixed arm 0.489. Noisy feedback: 0.851, rule 0.629, best fixed arm 0.835. Arbitrary rules: 0.525, rule 0.508, best fixed arm 0.570 (it loses here: there is nothing to learn, so it chases noise). |

| E11 rhythm | Beats equal to the creator's ideal on held-out videos, 4 creator types x 6 seeds: RLSVI 0.859, bandit on the same features 0.870, fixed rule 0.480. Whole-video rating: +0.04, +0.00, -0.41. RLSVI and the bandit are effectively tied. |

## Likely questions
- **Does RLSVI beat the bandit?** Not in E11: they tie, and both beat a fixed rule by a wide margin. These creators react to state both agents can see, so a bandit with history features captures most of it. Say that plainly: the rhythm decision is sequential by construction, but E11 does not show that the sequential method pays off yet.
- **Is this RL?** Yes, mostly the contextual-bandit case: it acts, sees feedback only on what it did, explores by Thompson sampling and learns online per creator. The target-length planner is the sequential part: an MDP with posterior sampling, known transitions and a learned reward. Do not call the whole system deep or full sequential RL.
- **Why RL, not classification?** Partial feedback, exploration, online per-user learning under drift, choosing what to ask, counterfactual evaluation. Binary keep/cut alone is close to supervised learning: say so first.
- **Why dynamic programming, not PPO?** Transitions are known and deterministic and the horizon is short, so exact backward induction is optimal for the given reward. PPO would approximate it from samples.
- **Did anything go wrong?** Yes, and it is a good story. With history features in the MDP state, the solver found "cut everything" optimal, because the linear model extrapolates into states it never saw (model exploitation). The reward now holds those features at the bandit's own trajectory. The cost: the reward no longer reacts to the plan's own history.
- **Does exploration harm the edit?** Random epsilon flips are off for keep/cut in the product (safe exploration), since a flip deletes a sentence. Thompson sampling still explores where the posterior is unsure. Consequence: for product logs, a phrase arm Thompson sampling never picks has no support, which weakens OPE for keep/cut policies far from the logging one. E9 itself ran with the epsilon floor on.
- **One click generalising too far?** With lambda 10, one rejected KEEP flipped 65% of kept phrases on the next video. lambda 30 for keep/cut: 8%, for a small learning cost (0.902 to 0.898 after 4 videos; 6 personas x 10 seeds).

## Limits to state honestly
- Simulated creators and TTS stress clips, plus one real English clip. The personas were written by us, and the priors are trained on them, so the simulator favours this model family.
- E9: intervals now use Student's t over videos with a small-sample factor, which raised coverage from 71-78%, but they still fall short of the nominal 95%: DR 89%, SNDR 86%, IPS and SNIPS 72%, DM 48%. Quote DR. Effective sample size is 15% of the logs because LinTS gets confident fast. In the simulator DM has the lowest RMSE (0.068) but poor coverage: a real bias-variance trade-off, and DM's win depends on the simulator being close to linear.
- Real logs are too few for OPE conclusions (24 rated suggestions in the current database): show `app.ope.run` as the method.
- English only. Other languages get a warning banner, not a good edit. The check is deliberately cautious, so a low-confidence detection shows no warning.
- Whisper can clean up stutters before we see them. wav2vec2's own reading now restores exact repeats and hesitations it hears clearly (the "I I" on the stress clip), but it can miss some.
- A filler touching its neighbours is cut only where the audio dips at both edges. On the stress clip 10 of 11 are cut; "basically" stays, as does "So that" on the real clip. They can be deleted in the transcript.
- Checked by re-transcribing the edited audio (stress clip: exactly the expected words), not by human listening.
- Beam size 10 fixed a misread line on the real clip ("this is the entity of mentors" -> "send it to your mentors"), chosen from that one clip. Beam 5 occasionally gave different text between runs, cause not found; do not promise identical transcripts.
- The speaker's name is transcribed "Taran" on the real clip. Names can be wrong: fix them by double-clicking the caption word.
- Hook selection picks the on-screen opening title; it does not move footage to the front.
- Lines spoken to the crew rather than the viewer are not detected, so they stay in the first cut.
- A target is a limit: a clip whose good edit is shorter stays shorter (the 30 s target on the stress clip gives ~19 s).
- The style sentence names only a leaning ("early guess, leaning Storyteller") until 8 reviewed decisions, then shows percentages. The percentages are still sharp by design; the bandit's prior moves far more slowly (tempered).
- First-video zooms are rare on purpose: the prior averages styles, most of which never zoom. Use "More zooms".
- The planner's reward ignores its own history features, on purpose: letting them follow the plan makes it cut everything. No experiment shows a gain from reacting to them, so it stays this way.
