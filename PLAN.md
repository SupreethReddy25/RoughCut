# RoughCut: Project Design & Implementation Plan (v3)

**Personalized short-form video editing using human-in-the-loop reinforcement learning**

Team: M. Rushi Varun Reddy · P. Supreeth Reddy · B. Vijaysimha Reddy

This document is the source of truth for building RoughCut. It is written so each phase in section 14 can be handed to a coding assistant (Sonnet) one at a time.

---

## 0. TL;DR

**Core purpose:** a talking-head creator should be able to go from raw clip to a **post-ready Reel without opening CapCut or any other simple editor**. Over time it should take *less* effort than a simple editor, because RoughCut learns how that creator edits.

- **Input:** one or more raw talking-head clips (30 s to 5 min total) with speech: vlogs, tutorials, opinions, "talking to camera" content.
- **Post-ready output:**
  - jump-cut edit with silences, retakes and filler words removed
  - face-centered 9:16 reframe
  - punch-in zooms
  - **animated word-by-word captions** with highlighted keywords
  - **cleaned, loudness-normalized audio**
  - optional **background music with auto-ducking**
  - text overlays (hook title)
  - platform-ready 1080×1920 MP4 + SRT
- **Two ways to edit, both feed the learner:**
  1. **AI suggestions** (pre-applied; accept / reject / override, ordered by the agent's uncertainty).
  2. **Manual editing** like a simple editor: Descript-style **transcript editing** (delete words to cut them), timeline, caption text fixes, text tool, music picker. Every manual edit that corresponds to a learnable decision is converted into feedback automatically.
- **Learned decisions (the RL part):**
  - phrase keep/cut
  - pause tightening
  - filler-word removal
  - punch-in zoom
  - caption keyword emphasis
  - (stretch) hook selection
- **Learning layers:**
  1. **Contextual bandits (Linear Thompson Sampling)** with a style-mixture prior. This is the core per-creator learner.
  2. (stretch) **Neural-Linear** shared representation.
  3. (stretch) **PPO on a learned reward model** for whole-edit planning (RLHF-style).
- **Autopilot:** once the agent is trusted, batch-upload clips and get "Ready to post" edits. Only low-confidence items are flagged.
- **Evaluation:**
  - simulated creator personas (learning curves, regret, cross-persona)
  - off-policy evaluation on real logs
  - a **head-to-head study vs a simple editor**: time-to-post, blind quality ratings, usability score. This directly tests the core purpose.

---

## 1. Core purpose: replacing a simple editor (v3)

### 1.1 What a creator actually does in a simple editor for a talking-head Reel

| # | Task in CapCut/InShot-style editor | RoughCut v3 | How |
|---|---|---|---|
| 1 | Trim dead air / long pauses | **Learned (D2)** | pause tighten/remove per gap |
| 2 | Cut mistakes, restarts, rambling | **Learned (D1)** + transcript editing | retake/redundancy/off-topic features |
| 3 | Remove "um / uh / like" | **Learned (D4)** + transcript editing | per filler-word decision |
| 4 | Reframe landscape → 9:16 on the face | **Automatic** + manual drag | per-clip face-centered crop |
| 5 | Punch-in zooms for emphasis | **Learned (D3)** + manual | 1.15× / 1.30× per kept phrase |
| 6 | Auto captions, word-by-word animated | **Automatic** (Whisper words) + style presets | captions remapped onto the output timeline |
| 7 | Highlight key words in captions | **Learned (D5)** | per caption line: which word (or none) gets accent color |
| 8 | Fix caption typos | **Manual** | edit words in transcript |
| 9 | Audio cleanup + consistent loudness | **Automatic** | high-pass, light denoise, loudnorm −14 LUFS |
| 10 | Background music, ducked under voice | **Manual pick, automatic ducking**; level remembered | bundled CC0 tracks + upload; sidechain ducking |
| 11 | Title / hook text on screen | **Manual** text tool (+ suggested from first sentence) | text overlays with style presets |
| 12 | Speed up slightly (1.1×) | **Setting**, default remembered per user | global speed |
| 13 | Join several takes into one video | **Automatic** | multi-clip upload, concatenated at ingest |
| 14 | Check it fits Reels/Shorts UI | **Automatic** | safe-zone overlay in preview, caption placement avoids UI |
| 15 | Export 1080×1920 for posting | **Automatic** | MP4 + SRT; platform presets |
| 16 | Come back later and continue | **Automatic** | every edit is saved server-side (projects list) |
| 17 | Undo mistakes | **Manual** | unified undo/redo over all edit events |
| 18 | Move best line to the start (hook) | **Learned (D6), Could** | hook selection |

**Deliberately not matched (rabbit holes for this semester):** transitions/effects library, stickers/GIFs, green screen, keyframe animation, multi-track B-roll, beat-synced music cuts, color grading, AI voice, stock footage, auto-generated visuals.

**Honesty rule for the learning claims:** settings like caption style, music level, and speed are **remembered defaults** (last used / running average), *not* RL. Say so in the report. The RL claims apply to D1–D6 only.

### 1.2 What makes it *better* than a simple editor (the pitch)
1. **Video 1:** about as fast as a simple editor's auto-features (auto captions + silence cut), plus transcript editing.
2. **Video 5+:** most suggestions already match your style, so you only check the "Needs your eye" queue.
3. **Autopilot:** trusted users batch-process clips and post with a handful of corrections.

Section 9, E11/E12, measures exactly this.

---

## 2. Version history (why things are the way they are)

**v2 research upgrades (kept):**

| # | Upgrade | Weakness it fixes | Priority |
|---|---|---|---|
| U1 | Log context + propensity for every suggestion | Nothing could be re-evaluated offline later | Must |
| U2 | "Why?" explanation per suggestion | Users can't tell what the model is doing | Must |
| U3 | Undo (policy recomputed from the log) | Mis-clicks corrupt online learning | Must |
| U4 | Semantic phrase features (local sentence embeddings) | Can't tell an important sentence from rambling | Should |
| U5 | Style-mixture prior | Cold start | Should |
| U6 | Uncertainty-driven review queue + "seen" tracking | Wasted review effort; noisy implicit accepts | Should |
| U7 | Off-policy evaluation (IPS / SNIPS / DR) | Real-user data can only evaluate the deployed algorithm | Should |
| U8 | Neural-Linear bandit | Linear features cap what can be learned | Could |
| U9 | Pairwise pacing comparisons (Bradley–Terry) | PPO's global reward was hand-set | Could |
| U10 | Hook selection | Standard Reels technique | Could |
| U11 | Optional LLM phrase scoring | Richer semantics | Could |

**v3 product upgrades (new), all aimed at replacing a simple editor:**

| # | Upgrade | Priority |
|---|---|---|
| P1 | **Transcript editor** (Descript-style word-level cut/restore, caption text fixes); manual edits → feedback | Must |
| P2 | **Animated captions** with style presets, safe-zone placement, preview = export parity | Must |
| P3 | **Audio enhancement** (high-pass, denoise, loudness normalization) | Must |
| P4 | **Per-clip face reframing** (+ manual crop drag) | Must |
| P5 | **Projects / autosave / unified undo-redo**, export presets + SRT | Must |
| P6 | **New learned decision D4: filler-word removal** | Should (rank 1) |
| P7 | **New learned decision D5: caption keyword emphasis** | Should (rank 2) |
| P8 | **Background music + auto-ducking** | Should (rank 3) |
| P9 | **Text overlays** (hook title etc.) | Should (rank 4) |
| P10 | **Autopilot + batch upload** with trust score | Should (rank 5) |
| P11 | **Multi-clip join** at ingest | Should (rank 6) |
| P12 | **Global nudges** ("Snappier / Calmer", "More / fewer zooms") + global speed | Should (rank 7) |
| P13 | Cover image export (frame + title) | Could |

**Deliberately not added (anywhere):** end-to-end deep RL on raw pixels/audio; generative video; an LLM "agent" that edits the video itself (it would replace, not strengthen, the RL contribution); B-roll/beat-sync/multi-cam; real authentication; mobile app.

---

## 3. Honest assessment

| Question | Answer |
|---|---|
| Realistic for 3 people in one semester? | Yes, **for talking-head / speech-driven content only**. Speech gives reliable, cheap structure (words, pauses, retakes, fillers). v3 adds real product work (captions, audio, transcript editor). That's why section 17 has a **ranked** Should list: cut from the bottom if time runs short. |
| Does the RL part make sense? | Yes: multi-option actions with partial feedback, an action-feature bandit (emphasis), exploration, non-stationary per-user preferences, choosing what to ask the human, off-policy evaluation, and sequential structure. Caveat: binary keep/cut is close to supervised learning. See section 13. |
| Biggest technical risks | Media plumbing (accurate cuts, filler-word cuts, preview = export parity for captions/music), not the RL math. See section 18. |

---

## 4. Scope

### In scope
- Projects: create, upload 1 to 5 clips (mp4/mov, ≤ 5 min total, ≤ 500 MB total), tag the video type (`tutorial | vlog | opinion | other`), set a target duration (optional).
- Automatic analysis: transcript with word timings, pauses, fillers, retakes, loudness, face track, semantic redundancy/off-topic scores.
- Learned decisions D1–D5 (D6 hook = Could).
- Editor:
  - 9:16 preview with captions, text, zoom, music, and safe-zone overlay
  - transcript editor
  - timeline
  - review queue with "Why?"
  - caption style / text / music / speed panels
  - unified undo/redo, keyboard shortcuts
- Per-user online learning, persisted per profile, with a full event log.
- Profile page: learned preferences in plain language, style mixture, learning + effort curves, trust score.
- Autopilot batch mode.
- Export: 1080×1920 H.264 MP4 + SRT (+ cover PNG, Could).
- Deployed via Docker.

### Out of scope
See 1.1 "deliberately not matched" and section 2 "deliberately not added".

---

## 5. System architecture

```
┌──────────────────────── Frontend (React + Vite + TS + Tailwind) ─────────────────────────┐
│ Projects → Upload → Editor:                                                              │
│   [9:16 Preview: video+captions+text+zoom+music, safe zones] [Review queue + Why?]       │
│   [Transcript editor]  [Timeline]  [Style panels: captions | text | music | speed]       │
│ → Profile / "What I learned" → Autopilot (batch)                                         │
│ Preview renders the EDL in the browser (skip ranges, CSS zoom/crop, HTML captions,       │
│ Web Audio music ducking). No server render while editing.                                │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │ REST (JSON): every edit = an event → new EDL
┌───────────────────────────────────────────▼──────────────────────────────────────────────┐
│ Backend (FastAPI, Python 3.11)                                                           │
│  api/        projects, videos, suggestions, events(+undo/redo), render, profile, autopilot│
│  pipeline/   ingest(+multi-clip join) → transcribe → pauses → segment → fillers →        │
│              features (audio, text, semantic, vision/face-track)                         │
│  policy/     LinTS (disjoint + action-feature), LinUCB, ε-greedy, Rule, style prior,     │
│              propensity, explain, (NeuralLinear), event log → sufficient statistics      │
│  session.py  left-to-right suggestion generation (shared with the simulator)            │
│  edl/        build EDL (cuts, crops, zooms, captions remap, texts, audio) — ONLY here    │
│  render/     captions→ASS, audio chain, music ducking, ffmpeg graph, SRT, cover          │
│  rl/         EditEnv + PPO, Bradley–Terry pacing reward                                  │
│  ope/        IPS / SNIPS / DR                                                            │
│  jobs        ThreadPoolExecutor background jobs                                          │
│  storage     SQLite (SQLModel) + files on disk                                           │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │ same feature cache (units.json)
┌───────────────────────────────────────────▼──────────────────────────────────────────────┐
│ sim/  persona simulator + experiment scripts (headless) → plots for the report           │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Principle 1: the EDL is the single contract.** All decisions and manual edits produce an EDL. The browser preview and the ffmpeg renderer both consume it, so what you see is what you export. **The EDL is built only in Python.** The frontend sends an event and gets back the new EDL (~50 ms), so there is one implementation of the timing math.

```json
{
  "video_id": "v_12",
  "edl_version": 3,
  "output": {"width": 1080, "height": 1920, "fps": 30, "speed": 1.0},
  "clips": [
    {"src_start": 0.84, "src_end": 3.12, "zoom": 1.0,  "crop_center": {"x": 0.52, "y": 0.41}, "unit_ids": [0]},
    {"src_start": 3.32, "src_end": 6.90, "zoom": 1.15, "crop_center": {"x": 0.50, "y": 0.40}, "unit_ids": [2]}
  ],
  "captions": {
    "enabled": true, "style_id": "bold_pop", "position": "lower_third",
    "lines": [
      {"t0": 0.00, "t1": 1.10, "words": [
        {"t0": 0.00, "t1": 0.32, "text": "Three"},
        {"t0": 0.32, "t1": 0.70, "text": "mistakes", "emphasis": true},
        {"t0": 0.70, "t1": 1.10, "text": "beginners"}]}
    ]
  },
  "texts": [{"id": "t1", "text": "3 editing mistakes", "t0": 0.0, "t1": 3.0, "position": "top", "style_id": "title"}],
  "audio": {"enhance": true, "target_lufs": -14,
            "music": {"track_id": "cc0_lofi_02", "gain_db": -18, "duck_db": -10}},
  "duration": 5.86
}
```
All caption/text times are in **output time** (after cuts and speed). Source→output remapping happens in `edl/build.py`.

**Principle 2: the event log is the source of truth.** Every suggestion stores its exact context vector, action, propensity, and policy version. Every user action (feedback, manual edit, setting change) is an appended event. The policy state and the current EDL are **caches recomputable from the log**. That gives undo/redo, prior swaps, neural retraining, and off-policy evaluation for free.

---

## 6. Problem formulation

### 6.1 Units

- **Speech unit (phrase):** a run of words between pauses ≥ 0.25 s, split further at sentence punctuation; max ~6 s.
- **Pause unit:** a gap between consecutive speech units (word gap ≥ 0.15 s, refined by an energy threshold). A **take boundary** (between joined clips) is always a pause unit with `is_take_boundary = 1`.
- **Filler token:** a word inside a phrase matching the filler lexicon (6.3, D4).
- **Caption line:** a group of ≤ 4 kept words / ≤ 1.4 s of output, broken at pauses and punctuation (built after cuts).

A 2-minute raw clip gives roughly 30 phrases, 30 pauses, 20 zooms, 10 to 25 filler tokens, and 30 to 40 caption lines. That's **~130 decisions per video**, so **5 videos ≈ 650 feedback signals**.

### 6.2 Decision types

| Decision | Applies to | Actions | Bandit form | Feedback |
|---|---|---|---|---|
| **D1 Phrase** | every speech unit | `KEEP`, `CUT` | disjoint LinTS | binary (reject ⇒ other is correct) |
| **D2 Pause** | every pause unit | `KEEP`, `TIGHT_40`, `TIGHT_15`, `REMOVE` | disjoint LinTS | **multi-arm, partial** |
| **D3 Zoom** | every kept phrase | `NONE`, `Z115`, `Z130` | disjoint LinTS | **multi-arm, partial** |
| **D4 Filler word** (v3) | every filler token in a kept phrase | `KEEP`, `REMOVE` | disjoint LinTS | binary |
| **D5 Caption emphasis** (v3) | every caption line | `NONE` or highlight candidate word *k* (≤ 3 candidates) | **action-feature LinTS** (shared θ over candidate features) | **slate / partial** |
| **D6 Hook** (Could) | once per video | `NONE` or move candidate phrase *k* ∈ top-3 to the start | action-feature LinTS | slate, 1 per video |

**Processing order (sequential features):** left to right. Per position: pause → phrase → (if kept) its filler tokens → zoom. After all units: build caption lines from kept words → D5 per line → (D6) hook.

### 6.3 Context features

Features are small (≈ 10–20 dims), normalized, and human-readable. Thresholdy continuous features are also **binned (one-hot)**. Every decision type gets a **video-type one-hot** and a **bias**.

**D1 Phrase**
- **Duration and delivery:** `duration`, `speech_rate`, `avg_word_conf`, `loudness_z`
- **Fillers:** `filler_ratio`, `starts_with_filler`, `nonword_voiced_ratio` (voiced audio with no transcribed words)
- **Retakes and incomplete lines:** `retake_score` (rapidfuzz vs the next 1–3 phrases), `is_earlier_take`, `is_incomplete`
- **Position and framing:** `position`, `is_first`, `is_last`, `face_present`
- **Semantics (U4):** `semantic_redundancy` (max cosine to other phrases), `topic_centrality` (cosine to the video mean embedding), `llm_importance` (U11, optional)
- **History:** `hist_output_ratio`, `hist_since_last_cut`

**D2 Pause**
`gap_duration` (log), gap bins [<0.3, 0.3–0.6, 0.6–1.0, 1.0–2.0, >2.0], `after_sentence_end`, `before_new_sentence`, `gap_rms`, `prev_has_filler`, `is_take_boundary`, `position`, `hist_output_ratio`.

**D3 Zoom**
`duration`, `loudness_z`, `speech_rate`, `has_emphasis_punct`, `is_first`, `face_size`, `topic_centrality`, `hist_since_last_zoom` bins [<3, 3–6, 6–10, >10 s], `hist_prev_zoomed`.

**D4 Filler word (v3)**
- **Token and position:** token one-hot [um, uh, like, you know, so, basically, actually, i mean, other], position in phrase [start, mid, end]
- **Timing:** `token_duration`, `gap_before`, `gap_after` (clean boundaries are safer to cut), `word_conf`
- **Ambiguity:** `likely_verb_like` ("like" preceded by I/you/we/they/would/don't/do/to, i.e. probably not a filler)
- **Delivery:** `loudness_z`

**D5 Caption emphasis (v3)**
- Features per **candidate word**:
  - `is_number`
  - `is_capitalized_mid_sentence` (proper noun)
  - `word_len`
  - `tfidf_in_transcript` (topic keyword)
  - `word_loudness_z` (vocal stress)
  - `position_in_line`
  - `is_last_in_sentence`
  - `is_negation` (no/never/not)
- `NONE` arm features: line length, `line_has_number`, bias.
- Candidates are the top-3 non-stopword words by a simple heuristic score (`tfidf + loudness`).
- One shared weight vector θ scores every candidate (action-feature linear bandit), so learning transfers across words.

**D6 Hook (Could)**
`topic_centrality`, `is_question`, `loudness_z`, `llm_hook_score`, `position`, `duration`.

**Semantic implementation (U4):** `sentence-transformers/all-MiniLM-L6-v2` embeds phrases at ingest. Raw embeddings go in `embeddings.npy` (for U8). The linear model only sees the derived scalars, so it stays interpretable.

### 6.4 Reward mapping: suggestions *and* manual edits

| Event | Update |
|---|---|
| Accept (explicit) | proposed arm r = +1 |
| Reject on binary (D1, D4) | proposed r = −1; other arm r = +1 |
| Reject on multi-arm (D2, D3, D5) without an alternative | proposed r = −1 only |
| Override to a′ (via panel, timeline, or transcript) | proposed r = −1; a′ r = +1 |
| Untouched at Finalize, unit was **seen** | proposed r = +0.5 |
| Untouched, never seen | no update |
| Undo / redo | mark the event undone / re-done → recompute from the log |

**How manual edits map to decisions (P1).** This is what lets a simple-editor workflow still train the agent:

| Manual edit | Mapped to |
|---|---|
| Delete or restore a whole phrase in the transcript or on the timeline | D1 override |
| Delete a word range covering ≥ 70% of a phrase's speech | D1 override → CUT (the rest of the phrase is cut too, shown to the user) |
| Delete or restore a single filler token | D4 override |
| Delete a non-filler word range < 70% of a phrase | `fine_cut` event: applied to the EDL, **logged but not learned** (out of the action space) |
| Change a pause option, or drag a pause shorter/longer | D2 override (snapped to the nearest option) |
| Change zoom on a clip | D3 override |
| Click a different caption word to emphasize / clear emphasis | D5 override |
| Boundary nudge (±0.05 s), crop drag, caption text fix, text overlay, music, speed | edit-only events (not learned; settings feed remembered defaults) |
| Global nudge "Snappier / Calmer" etc. (P12) | one `nudge` event: adds a score offset to the relevant arms *for this video*, re-runs suggestions (new rows, `source = nudge`, excluded from OPE). The user's later accept/reject on those still trains normally. |

Weights live in `policy/reward.py` so they can be ablated.

---

## 7. Learning algorithms

### 7.1 Core: Linear Thompson Sampling with prior + discounting

**Disjoint form** (D1–D4): per arm *a*, expected reward ≈ θₐᵀx.
**Action-feature form** (D5, D6): one θ; the score of candidate *k* = θᵀx_k. Same math with one A, b.

Prior and user data are separate sufficient statistics:

```
prior:      A0 = λ·I,  b0 = λ·θ_prior                 # θ_prior from 7.2
data:       A_d = 0,   b_d = 0
posterior:  A = A0 + A_d ;  θ̂ = A⁻¹(b0 + b_d) ;  Σ = v²·A⁻¹

choose(x):  draw θ̃ ~ N(θ̂, Σ); a* = argmax score; with prob ε_floor = 0.02 pick uniformly
            explored = (a* ≠ argmax under θ̂)
update(x, r, w):   A_d ← γ·A_d + w·xxᵀ ;  b_d ← γ·b_d + w·r·x       # discounting = drift
recompute_from_log(user): replay non-undone events in order
```

Defaults: `λ = 2.0`, `v = 0.5`, `γ = 0.995`. Tune on the simulator.

**Propensity + confidence (U1, U6):** run M = 200 Thompson draws; `p(a) = (1−ε_floor)·freq_argmax(a) + ε_floor/K`. Store `propensity = p(a*)`. Use the same value as `confidence` for the review queue.

**Baselines** (same interface `choose → (action, propensity, confidence)`, `update`):
- `RulePolicy`: fixed thresholds, i.e. what simple editors' one-click features do. Cut pauses > 0.5 s, cut earlier takes, remove um/uh only, no zoom, emphasize the longest word.
- `GlobalPolicy`: generic prior, never updated.
- `EpsGreedyLinear` (ε = 0.1)
- `LinUCB` (α = 1.0)
- `LinTS` (ours)

### 7.2 Cold start: style-mixture prior (U5)
- **Style prototypes:** θ⁽¹⁾…θ⁽ᴷ⁾ (one full set of weights per style).
  - Initially: ridge fits on simulated feedback, one per persona.
  - After the real study: also k-means over real users' posterior means.
  - Plus θ⁽⁰⁾ = pooled generic.
- **Style posterior:** `log w_k ∝ −Σ (r − score⁽ᵏ⁾(x, a))² / (2σ²)`, σ = 0.7.
- **Prior mean:** `θ_prior = Σ w_k θ⁽ᵏ⁾`. Update while the user has < 200 events, then freeze it.
- **UI:** "Your style so far: 70% Snappy · 25% Storyteller."

### 7.3 Uncertainty-driven review queue (U6)
- **Priority** = `1 − confidence`.
- **Panel:**
  - **"Needs your eye"**: confidence < 0.7, sorted by priority.
  - **"Confident edits (N), spot-check"**: the rest, collapsed.
- `J` jumps to the next item in the queue. Timeline/transcript items show a confidence dot.
- Live re-scoring after every feedback re-sorts the queue.

### 7.4 Off-policy evaluation (U7)
**Estimators** over explicit-feedback events with logged propensities:
- IPS
- SNIPS
- DR (q̂ = cross-fitted ridge)

Bootstrap CIs. Nudge-sourced and manual-only events are excluded.

**Validation (E9):** check the estimators in the simulator first, then rank all policies on the real logs.

### 7.5 Neural-Linear bandit (U8, Could)
- A shared MLP body φ(x_raw) → ℝ⁸, trained offline on pooled users + personas.
- Per-user LinTS head on `[x_hand, φ]`.
- Heads are rebuilt from the log after each body retrain.
- Evaluated in E10.

### 7.6 Sequential layer: PPO on a learned reward (Could; promote if the course emphasizes MDPs)
- **`EditEnv` (gymnasium):**
  - **Step:** one phrase + its preceding pause.
  - **Action:** `MultiDiscrete([4, 2, 3])` = (pause, phrase, zoom).
  - **Per-step reward:** the user's bandit posterior means.
  - **Terminal reward:** `−β·|duration − target|/target` + `η·wᵀs` (U9 Bradley–Terry rhythm score from "Which feels better?" comparisons) − a penalty for clips < 0.4 s.
- **Training:** SB3 PPO, ~50k steps per user (1–3 min CPU), retrained after each finalize.
- **Usage:** used when a target duration is set and the user has ≥ 2 finalized videos.
- D4/D5 stay with the bandit.

### 7.7 Explanations (U2)
- **Shown:** the top-3 feature contributions relative to the runner-up arm, rendered as sentence templates ("This 'like' sits between clean gaps. You usually remove those."), plus the confidence.
- **D5:** "You tend to highlight numbers."

---

## 8. Product features spec (v3)

### 8.1 Transcript editor (P1, Must)
- **Layout:**
  - The transcript is shown as phrases; kept words are normal, cut words are ~~struck~~ and greyed.
  - Filler tokens are underlined (dotted); removed fillers are struck.
  - Pause markers sit inline (`⏸ 0.8s`) and are clickable → the pause options popover.
- **Interactions:**
  - Click a word → seek.
  - Select a range → `Delete`/`Backspace` cuts it; select struck words → `Restore`.
  - Double-click a word → edit its caption text (fixes typos; timings unchanged).
- **Sync with preview:** the currently playing word is highlighted (karaoke) to keep orientation.
- **Events:** every action becomes an event (section 6.4 mapping). The backend returns the new EDL + re-scored suggestions.
- **Implementation:** plain React spans (no rich-text editor library); selection via `window.getSelection()` mapped to word ids through `data-word-id` attributes.

### 8.2 Captions (P2, Must)
- **Source:** kept Whisper words remapped to output time; lines grouped per 6.1.
- **Style presets** in `render/caption_styles.json`, 4 of them:
  - `bold_pop`: Montserrat ExtraBold, white + black outline, emphasis = yellow, active word scales up 1.1×
  - `clean`: Inter SemiBold, white on a translucent box
  - `karaoke`: words fill with color as spoken
  - `minimal`: small lowercase, no outline
- **Positions:** `lower_third` (default, ≈ 68% height, above platform UI), `center`, `top`.
- **Preview:** a React `CaptionOverlay` driven by the same JSON (CSS from style fields), placed over the video in the 9:16 frame.
- **Export:** generate an **ASS** subtitle file from the same JSON (per-word events / `\k` karaoke tags), burned in with ffmpeg `subtitles=...:fontsdir=assets/fonts`.
- **Parity:**
  - Fonts: the same OFL font files are served to the browser via `@font-face` and given to libass.
  - Sizes: defined in **px at 1080×1920**; the preview scales them by `container_width/1080`.
- **SRT** export alongside the MP4.
- Caption on/off toggle; style choice is a remembered default.

### 8.3 Audio enhancement (P3, Must)
- **Chain on export:**
  - `highpass=f=80`
  - `afftdn=nf=-25` (light; skip if the measured noise floor is already low)
  - **two-pass** `loudnorm=I=-14:TP=-1.5:LRA=11`
  - 8 ms `afade` at every cut
- **Preview:** a toggle shows "Enhanced audio is applied on export". Optionally preview with a Web Audio high-pass + compressor approximation (not required).

### 8.4 Reframing (P4, Must)
- **Face tracking:** YuNet at 5 fps on the proxy → face center track.
- **Crop centers:**
  - per clip `crop_center` = median face center within the clip
  - clamp so the crop stays in-frame
  - if no face, use the previous clip's center
  - cuts hide the jumps, so no smoothing animation is needed
- **Portrait sources:** no crop; zoom origin = face center.
- **Manual:** drag the video inside the 9:16 frame for the selected clip → `crop_override` event (edit-only).

### 8.5 Projects, autosave, undo/redo, export (P5, Must)
- **Projects list:** thumbnail, status (`processing | ready | edited | exported`), last edited. Everything is server-side, so closing the tab loses nothing.
- **Unified undo/redo:** a stack over the user's events for this video (Ctrl+Z / Ctrl+Shift+Z). Backend `undo` / `redo` endpoints flip `undone` and recompute.
- **Export presets** (`render/presets.json`): Reels / Shorts / TikTok are all 1080×1920 @ 30 fps, H.264 High CRF 20, AAC 48 kHz 192 kbps. Each preset has a max-duration warning value kept in config; verify current platform limits when writing it.
- **Safe-zone overlay:** a toggle in the preview showing the platform UI regions (top bar, right action rail, bottom caption area) from the preset.

### 8.6 Background music + ducking (P8, Should #3)
- **Library:** 8–10 bundled **CC0 / royalty-free** tracks (verify each license; record it in `assets/music/LICENSES.md`) + user upload (mp3/wav ≤ 10 MB).
- **Controls:** track picker with a 10 s preview, music volume (dB), duck amount, fade in/out (1 s).
- **Export:** music looped/trimmed to the output duration → `sidechaincompress` keyed by the voice track (threshold 0.03, ratio 8, attack 20 ms, release 300 ms) → `amix` → loudnorm on the final mix.
- **Preview:** a second `<audio>` through Web Audio `GainNode`, with gain automation scheduled from the EDL's speech regions (an approximation of the ducking).
- **Remembered defaults:** last track style and level.

### 8.7 Text overlays (P9, Should #4)
- **Tool:** add/edit/delete text boxes: text, start/end (drag on the timeline text track), position (`top | center | custom y`), and a style preset (`title`, `label`, `callout`).
- **Suggested hook title:** on first open, prefill a `title` text with the first sentence (≤ 8 words) for 0–3 s, shown as a suggestion the user can delete (*not* learned in v3).
- **Render:** same pipeline as captions (ASS layer above captions) and the same overlay component in preview.

### 8.8 Autopilot + batch (P10, Should #5)
- **Trust score** = explicit-acceptance rate over the user's last 2 finalized videos (reviewed suggestions only). Shown on the profile.
- **Batch upload:** up to 5 projects at once; each is processed and auto-edited with the user's current policy (posterior mean, no exploration).
- **Per-project badge:**
  - **"Ready to post"** if expected corrections `Σ(1 − confidence)` < 3 *and* trust ≥ 0.85
  - otherwise **"Needs review (N)"**, which opens the editor with the queue pre-filtered
- **One-click export** for ready projects (captions/music/text use remembered defaults).
- **Learning:** exported-without-review projects produce **no** training signal (nothing was seen). Only reviewed items train.

### 8.9 Multi-clip join (P11, Should #6)
- **Upload:** multiple files into one project with a reorder list.
- **Ingest:** normalize each (fps 30, same resolution/rotation, 48 kHz audio), then concatenate into `source.mp4` with a `source_map`. Downstream code sees one source.
- **Take boundaries** are pause units with `is_take_boundary = 1`. Creators almost always remove them, so it learns fast.

### 8.10 Global nudges + speed (P12, Should #7)
- **Nudge buttons:**
  - "Snappier": +offset to `TIGHT_15`/`REMOVE`/`CUT`/filler `REMOVE`
  - "Calmer": the opposite
  - "More zooms" / "Fewer zooms"
- **Effect:** applied as score offsets for this video, then suggestions are re-run (6.4).
- **Speed:** 1.0 / 1.1 / 1.15 / 1.25×. Export uses `setpts` + `atempo` after cuts; caption times are scaled in the EDL; preview uses `playbackRate`. Remembered default.

### 8.11 Cover image (P13, Could)
Pick a frame from the preview, add the title text, export a 1080×1920 PNG (ffmpeg single-frame render with the ASS title).

---

## 9. Persona simulator

```python
class Persona:
    name: str
    def ideal_action(self, decision_type, item, history) -> action   # hidden ground truth
    def prefers(self, window_a_stats, window_b_stats) -> bool        # U9
    target_duration: float | None
    p_override: float = 0.6
    noise: float = 0.05
    review_budget: float = 1.0
```

| Persona | Pauses (D2) | Phrases (D1) | Fillers (D4) | Zoom (D3) | Emphasis (D5) |
|---|---|---|---|---|---|
| **Snappy** | REMOVE > 0.5 s, else TIGHT_15 | cut fillery, earlier takes, low-conf | remove all incl. "like" | Z115 on loud, ≥ 4 s apart | numbers, else loudest word |
| **Relaxed** | KEEP < 0.8 s, else TIGHT_40 | cut earlier takes only | remove um/uh only | never | none |
| **Storyteller** | TIGHT_40 after sentence end, REMOVE mid-sentence | cut phrase-start fillers | remove um/uh/"you know" | Z130 on ?/! and the first phrase | proper nouns / keywords |
| **Minimalist** (target 30 s) | REMOVE all | keep high-centrality until target | remove all | none | none |
| **Focused** | TIGHT_15 | cut tangents (centrality < 0.3) & repeats | remove um/uh/like | Z115 on central phrases | top TF-IDF keyword |
| **Blend** | 50/50 Snappy/Storyteller | ↔ | ↔ | ↔ | ↔ |

Variants: **drift** (Relaxed → Snappy at video 6) and **type-conditional** (Snappy on vlogs, Relaxed on tutorials).

The simulator also models **manual editing**: when a persona disagrees, with prob 0.5 it "edits in the transcript" instead of the panel. That exercises the manual → feedback mapping with the same math.

---

## 10. Experiments

Simulator experiments: 10 seeds, mean ± 95% CI.

| ID | Question | Setup | Metric |
|---|---|---|---|
| **E1** | Does it learn each creator? | every persona × {Rule, Global, ε-greedy, LinUCB, LinTS}, 10 videos, all decision types | **acceptance vs video index** (overall + per decision type); corrections/min; regret |
| **E2** | Does the style prior fix cold start? | {none, generic, style-mixture}; leave-one-persona-out | acceptance on videos 1–3; decisions to identify style |
| **E3** | Is it personalized? | train on P, evaluate on Q | cross-persona matrix |
| **E4** | Drift | Relaxed→Snappy at video 6; γ ∈ {1, 0.995, 0.98} | recovery curve |
| **E5** | Sequential RL (Could) | Minimalist: LinTS vs PPO (±U9) | acceptance + duration error |
| **E6** | Real users learn-over-time | 3 teammates (written style guides) + 3–5 volunteers, 6–8 videos each | acceptance + effort curves, survey |
| **E7** | Ablations | feature groups; implicit weight; seen-tracking; manual-edit mapping on/off | Δ acceptance |
| **E8** | Asking the right questions | review budget 10–100%; uncertainty vs random vs timeline | acceptance vs budget |
| **E9** | Trust OPE? | simulator truth vs IPS/SNIPS/DR; then rank policies on real logs | estimation error; ranking table |
| **E10** | Neural-Linear (Could) | vs linear, esp. Focused | curves; OPE |
| **E11** | **Does it replace a simple editor?** (core purpose) | 6–8 participants. Each edits the **same raw clip** in a simple editor they know (CapCut/InShot) and in RoughCut, order counterbalanced. Done at the participant's 1st RoughCut video and again after 5 RoughCut videos. Goal: "post-ready Reel with captions". | **time-to-export**; number of manual actions; **blind pairwise quality rating** by 5–10 raters ("which would you post?", win/tie/loss); **SUS** usability score; "tasks I had to leave the app for" checklist |
| **E12** | Autopilot | after ≥ 5 videos, each participant batch-processes 3 new clips | % "Ready to post" correct (participant agrees with ≤ 2 changes); corrections per clip |

**Headline claims to aim for (report whatever actually comes out):**
1. By video 5, acceptance ≥ 85% on simulator personas and clearly above the Rule baseline.
2. In E11, time-to-post with RoughCut after 5 videos is substantially lower than with the simple editor, with no loss in blind quality preference.

---

## 11. Data
- **Recordings:** each member records **15 raw clips** (45–150 s, phone, mix of tutorial/vlog/opinion) with natural mistakes: restarts, pauses, um/uh/like, a tangent or two. ~45 clips. 5 of them are **split into 2–3 takes** to test multi-clip join.
- **Consent** from everyone recorded or participating.
- **E11:** 2 shared raw clips used by all participants (same footage → fair comparison), plus their own clips for learning.
- **Logs as a dataset:** export events + suggestions after the study (CSV/Parquet).
- `data/` is gitignored; keep a shared drive.

---

## 12. Media pipeline details

| Step | Tool | Details / gotchas |
|---|---|---|
| Ingest / join | `ffmpeg`/`ffprobe` | Normalize each input: CFR 30 fps, honor rotation, 48 kHz. Concat → `source.mp4` + `source_map`. Make `proxy.mp4` (720p-ish, `-g 15`) and `audio.wav` (16 kHz mono). |
| Transcribe | `faster-whisper` `base.en` int8 CPU (`small.en` on GPU) | `word_timestamps=True`. Disfluency `initial_prompt="Umm, let me think, like, hmm... Okay, so, uh, here's what I, like, mean."`, since Whisper otherwise drops many um/uh. Back it up with `nonword_voiced_ratio`. |
| Pauses | numpy RMS (20 ms) | Word gaps ≥ 0.15 s, edges refined with a per-video percentile energy threshold. |
| Segment | custom | Split at pauses ≥ 0.25 s + punctuation; merge < 0.4 s; cap ~6 s. |
| Fillers | lexicon + rules | Token match + `likely_verb_like`. **Filler cut boundaries:** cut from the previous word end + 20 ms to the next word start − 20 ms. Skip the suggestion (default KEEP, confidence low) if the gap on either side is < 30 ms (co-articulated, so cutting would clip neighbors). |
| Retakes | `rapidfuzz` | token_set_ratio vs the next 1–3 phrases. |
| Semantics | MiniLM | Redundancy, centrality, `embeddings.npy`. TF-IDF over the transcript for D5. |
| Word stress | numpy | Per-word RMS z-score (D5 `word_loudness_z`). |
| Face track | OpenCV YuNet (ONNX in repo) | 5 fps on the proxy; per-clip median center; face size per phrase. |
| Features | numpy | `units.json` with `feature_version`. |
| Render | ffmpeg filter graph (`render/graph.py`) | Per clip: trim/atrim + afade 8 ms + crop (9:16 at crop_center, window shrunk by zoom) + scale 1080×1920 (crop from **source**, not proxy) → concat → speed → `subtitles` (ASS captions + texts) → audio chain (8.3) + music (8.6). For > ~80 clips, render segments then concat demuxer. |
| Cut padding | EDL builder | Pad speech edges by 40 ms. |

**Browser preview:**
- **Video:** the `<video>` proxy in a 9:16 frame, with a rAF loop that skips cut ranges. Per-clip CSS `transform: scale(zoom)` + translate for the crop center.
- **Overlays and audio:** `CaptionOverlay` + `TextOverlay` driven by output time, and Web Audio music ducking.
- **Speed:** `playbackRate`.
- **Seen tracking:** the loop reports seen units.

---

## 13. Viva defense: "Why is this RL and not classification?"
1. **Partial feedback + exploration:** D2/D3 are multi-arm and D5 is a slate choice. A rejection only says "not this", so Thompson Sampling must explore.
2. **The agent chooses what to ask about:** the uncertainty-driven review queue (E8).
3. **Online, per-user, non-stationary:** no dataset up front; it tracks drift (E4).
4. **Counterfactual evaluation:** propensity-logged OPE (E9).
5. **Sequential structure:** MDP + PPO on a reward learned from human feedback, incl. pairwise comparisons (RLHF structure; E5).
6. **Caveat to state yourself:** binary decisions (D1, D4) are close to online supervised learning. The RL-specific contributions are 1–5.
7. **Product claim vs research claim:** "replaces a simple editor" is measured by E11/E12 (HCI evaluation). "Learns the creator" is measured by E1–E9 (RL evaluation). Keep them separate in the report.

**Literature additions:**
- Riquelme et al. 2018 (Neural-Linear)
- Dudík, Langford & Li 2011 (DR OPE)
- Agrawal & Goyal 2013 (LinTS)
- Russo et al. 2018 (TS tutorial)
- Brooke 1996 (SUS)
- Descript / CapCut as product baselines (already in the Existing Systems study)

---

## 14. Implementation phases (hand to Sonnet one at a time)

Each phase ends with **acceptance criteria**. Don't start the next phase until they pass.

### Phase 0: Scaffolding (Week 1)
- Monorepo (section 15), `pyproject.toml`, ruff, pytest; Vite React TS + Tailwind.
- `schemas.py`: `UnitsFile`, `EDL` (v3, incl. captions/texts/audio), `Suggestion`, `Event`. Matching TS types.
- Bundle fonts (OFL) in `assets/fonts/`; ffmpeg PATH check; `/api/health`.
- **Accept:** tests run; the frontend calls health; the EDL schema round-trips JSON in both languages.

### Phase 1: Pipeline + render core (Weeks 2–3)
- `pipeline/`: ingest (incl. multi-clip join), transcribe, pauses, segment, fillers, features, semantic, vision (face track).
- `edl/build.py`: decisions → clips, crop centers, caption lines remapped to output time, speed.
- `render/`: `graph.py`, `ass.py` (captions + texts from styles), `audio.py` (enhance chain), `srt.py`.
- CLI: `python -m app.pipeline.run data/raw/a.mp4 [data/raw/b.mp4 ...] --type vlog`; `python -m app.render.run <edl.json>`.
- **Accept:**
  - On 3 real clips, units and filler tokens look right by eye.
  - A rule-based EDL renders 1080×1920 with captions burned in (style `bold_pop`), face-centered crop, no clicks/clipped words, and integrated loudness −14 ± 1 LUFS (measure with `ffmpeg -af ebur128`).
  - Rendered duration = EDL duration ± 0.1 s.
  - A 2-take project joins cleanly.

### Phase 2: Policies + simulator (Weeks 3–5)
- `policy/`: base, linear (disjoint + action-feature LinTS, LinUCB, ε-greedy), rule, reward (incl. the manual-edit mapping), log_replay, store, prior, explain.
- `sim/`: personas (all decision types, incl. manual-edit behavior), session; experiments E1, E3, E4.
- Tests:
  - LinTS converges (both forms).
  - Replay equals online.
  - Propensities are valid.
  - Each manual-edit → feedback mapping row of 6.4 is covered.
- **Accept:** the E1 plot shows LinTS beating Rule/Global by video 5 for every persona (overall and per decision type); E3 is diagonal-dominant. **Review 2 result.**

### Phase 3: Backend API (Weeks 4–5)
- SQLModel models (16.1), routes (16.2), jobs, Range serving.
- The event pipeline: event → reward mapping → policy update → EDL rebuild → re-score → response.
- Undo/redo, seen, live re-scoring.
- **Accept:** a TestClient script covers create project (2 clips) → ready → suggest → 10 feedbacks + 3 transcript edits → undo/redo (state matches) → finalize → render → download MP4 + SRT.

### Phase 4: Editor UI core (Weeks 4–7)
- **Pages:** Projects, Upload (multi-file + reorder, type, target), Editor, Profile (basic).
- **Editor:**
  - `PreviewPlayer` (clips, crop/zoom, `CaptionOverlay`, safe-zone toggle, seen reporting)
  - **`TranscriptEditor`** (8.1)
  - `Timeline` (clips, pauses, zoom badges, confidence dots, text track)
  - `ReviewPanel` + `WhyBox`
  - `CaptionStylePanel`
  - crop drag
  - unified undo/redo
- **Keyboard:** `A`, `R`, `1-4`, `J`, `Space`, `←/→`, `Del`, Ctrl+Z / Ctrl+Shift+Z.
- **Accept:**
  - A user can produce a captioned, reframed, cleaned Reel from a 2-minute clip **without leaving the app**.
  - Preview matches export (captions within 1 frame, same line breaks).
  - Every transcript edit shows up as the correct event type in the log.

### Phase 5: Product completeness (Weeks 6–8), in rank order (stop where time runs out)
1. D4 filler decisions surfaced in the UI (underline + queue).
2. D5 emphasis (click a caption word to set/clear).
3. Music + ducking (8.6).
4. Text overlays + suggested hook title (8.7).
5. Autopilot + batch + trust score (8.8).
6. Global nudges + speed (8.10).

**Accept:** each shipped item works in preview and export, and its events are logged.

### Phase 6: Style prior, review queue polish, profile (Weeks 6–8, member A)
- `policy/style.py`; profile page (preferences text, style mixture, learning/effort curves, trust score); experiments E2, E8.
- **Accept:** Snappy is identified (> 0.8) within ~30 decisions; E2 and E8 plots are produced.

### Phase 7: OPE (Week 8, member B)
- `ope/estimators.py`, `ope/run.py`; E9 in the simulator.
- **Accept:** SNIPS/DR within ±0.05 of the truth on the simulator, or report honestly.

### Phase 8: Deployment + studies (Weeks 8–10)
- Single Docker image (frontend built and served by FastAPI; ffmpeg; Whisper + MiniLM + fonts + music pre-bundled).
- Host on HF Spaces (Docker, free CPU) / Render / a college server. Export logs daily (ephemeral storage).
- Limits: ≤ 5 min total, ≤ 500 MB, one processing job at a time.
- Run **E6, E11, E12**, then E7 and E9 on the real logs.
- **Accept:** a public URL works; E11 data collected (timings, ratings, SUS); logs exported.

### Phase 9 (Could): PPO + pacing comparisons (E5), Neural-Linear (E10), hook D6, LLM scoring, cover image
Only after Phase 8's acceptance criteria pass, except PPO, which member A can start in Week 8 if ahead.

### Phase 10: Report + demo (Weeks 11–12)
- Report sections: the RL evaluation (E1–E10) and the product evaluation (E11–E12), kept separate.
- **3-minute demo:**
  1. Raw clip → messy first edit → corrections in the transcript + queue.
  2. Video 5: the "Needs your eye" queue is short and the style is detected.
  3. Autopilot batch → "Ready to post" → exported Reel playing on a phone.

---

## 15. Repository layout

```
RoughCut/
├── PLAN.md  README.md  Dockerfile  docker-compose.yml
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py  config.py  schemas.py  db.py  models.py  jobs.py
│   │   ├── api/          projects.py videos.py suggestions.py events.py profile.py autopilot.py admin.py
│   │   ├── pipeline/     ingest.py transcribe.py pauses.py segment.py fillers.py features.py
│   │   │                 semantic.py vision.py llm_score.py run.py
│   │   ├── policy/       base.py linear.py rule.py reward.py prior.py style.py log_replay.py
│   │   │                 store.py explain.py neural_linear.py
│   │   ├── session.py    # left-to-right suggestion generation (shared with sim)
│   │   ├── edl/          build.py remap.py
│   │   ├── render/       graph.py ass.py audio.py music.py srt.py cover.py run.py
│   │   │                 caption_styles.json text_styles.json presets.json
│   │   ├── rl/           edit_env.py train.py policy.py pacing_bt.py
│   │   └── ope/          estimators.py run.py
│   ├── sim/              personas.py session.py experiments/e1..e12
│   ├── scripts/          train_body.py build_prototypes.py export_logs.py
│   ├── assets/           face_detection_yunet.onnx  fonts/  music/ (+LICENSES.md)
│   └── tests/
├── frontend/src/
│   ├── api.ts  types.ts
│   ├── pages/            Projects Upload Editor Profile Autopilot
│   └── components/       PreviewPlayer CaptionOverlay TextOverlay SafeZones TranscriptEditor
│                         Timeline ReviewPanel WhyBox CaptionStylePanel TextTool MusicPanel
│                         NudgeBar StyleMixture LearningCurve ComparePacing
├── data/                 (gitignored) raw/ processed/ db.sqlite logs/
└── reports/figures/
```

**Dependencies:** fastapi, uvicorn, sqlmodel, python-multipart, faster-whisper, sentence-transformers, numpy, scipy, scikit-learn, rapidfuzz, opencv-python-headless, torch (CPU), gymnasium, stable-baselines3, matplotlib, pytest. Frontend: react, react-router, tailwindcss, recharts. System: ffmpeg built with libass (check `ffmpeg -filters | grep subtitles`).

---

## 16. Backend spec

### 16.1 Data model (SQLite via SQLModel)
```
User(id, name, created_at, defaults_json)          # remembered caption style, music level, speed, export preset
Project/Video(id, user_id, title, sources_json, source_map_json, duration, video_type, target_duration,
      status[uploaded|processing|ready|edited|exported|failed], feature_version, settings_json,
      autopilot bool, created_at, updated_at, error)
Unit(id, video_id, idx, kind[speech|pause|filler], parent_unit_id, start, end, text, features_json, seen)
Word(id, video_id, idx, unit_id, start, end, text_original, text_display, conf, is_filler)
Suggestion(id, video_id, target_id, decision_type[phrase|pause|zoom|filler|emphasis|hook],
           proposed_action, candidates_json, context_raw_json, propensity, confidence, explored,
           source[policy|nudge|autopilot], policy_name, policy_version, feature_version,
           superseded, created_at)
Event(id, user_id, video_id, kind[accept|reject|override|implicit|fine_cut|crop|caption_text|
      text_overlay|music|speed|nudge|setting], suggestion_id?, payload_json, reward?, weight?,
      undone, ts)
TextOverlay(id, video_id, text, t0, t1, position, style_id)
Comparison(id, user_id, video_id, stats_a_json, stats_b_json, choice, ts)
PolicyState(user_id, decision_type, algo, blob_path, n_events, version)     # cache only
StylePosterior(user_id, weights_json, frozen, updated_at)
Metric(user_id, video_id, video_index, n_decisions, n_reviewed, n_accepted, n_corrected,
       n_manual_actions, acceptance, seconds_active, exported_via_autopilot, ts)
StudySession(id, participant, condition[roughcut|simple_editor], clip_id, seconds, n_actions, ts)  # E11
```
`seconds_active` = time in the editor with focus (not wall clock), measured client-side.

### 16.2 API
| Method & path | Purpose |
|---|---|
| `GET/POST /api/users` | profiles (name only) |
| `GET/POST /api/projects` (multipart files[], order, type, target) | list / create → ingest job |
| `GET /api/projects/{id}` | status, units, words, suggestions, EDL, settings |
| `GET /api/projects/{id}/proxy` | proxy (Range) |
| `POST /api/projects/{id}/suggest` | run the policy → suggestions + EDL + queue |
| `POST /api/projects/{id}/events` `{kind, ...}` | **single entry point for all user actions**; returns the new EDL, re-scored suggestions, queue, style mixture |
| `POST /api/projects/{id}/undo`, `/redo` | flip the last event; recompute |
| `POST /api/projects/{id}/seen` | batch seen units |
| `POST /api/projects/{id}/finalize` | implicit accepts (seen only), metrics, PPO retrain, ≤ 2 comparisons |
| `POST /api/projects/{id}/render` `{preset}` → `GET .../download?fmt=mp4\|srt\|png` | export |
| `GET /api/music` / `POST /api/music` | library list / upload |
| `POST /api/autopilot` (multipart, multiple projects) / `GET /api/autopilot/{batch}` | batch create + status/badges |
| `GET /api/users/{id}/profile` | preferences text, style mixture, weights, stats, learning/effort curves, trust |
| `POST /api/users/{id}/reset` · `GET /export` · `POST /import` | profile management |
| `POST /api/study/sessions` · `GET /api/admin/logs.csv` | E11 timing entries; full logs |

**Live re-scoring:** after each learnable event, re-run the posterior mean on unreviewed items only; mark the old rows `superseded`; highlight the changes ("Adjusted 3 suggestions based on your last change").

### 16.3 Profile generation
- **Sentences** from top-weight features per decision, e.g.:
  - "You remove pauses longer than ~0.6 s."
  - "You keep 'like' but always cut 'um'."
  - "You highlight numbers in captions."
  - "You rarely zoom."
- **Charts and stats:** style mixture, stats (avg clip length, cuts/min, zooms/min, % kept, fillers removed/min), learning curve, effort curve (manual actions per minute of footage), trust score.

---

## 17. Team, timeline, priorities

### Team split
| Member | Owns |
|---|---|
| **A: RL/ML** | features (incl. filler, emphasis), all policies, style prior, explain, simulator, E1–E5, E7–E8, E10, PPO |
| **B: Backend/media** | ingest/join, transcribe, face track, EDL builder + remap, **render (captions ASS, audio, music, speed)**, API + event pipeline + undo/redo, OPE (E9), Docker/deploy |
| **C: Frontend/UX + studies** | Projects/Upload/Editor/Profile/Autopilot pages, **PreviewPlayer + overlays**, **TranscriptEditor**, Timeline, panels, keyboard, runs E6/E11/E12 |

Contracts (`schemas.py`: units, words, EDL v3, suggestion, event) are **frozen in Week 2**.

### Timeline (12 weeks)
| Week | Milestone |
|---|---|
| 1 | Scaffolding, dataset recorded (incl. multi-take clips), schemas, fonts/music licensed |
| 2–3 | Pipeline + render core (captions, audio, reframe) headless; PreviewPlayer + CaptionOverlay from a JSON EDL |
| 3–5 | Policies D1–D5 + simulator + E1/E3/E4 **(Review 2: "it learns")** |
| 4–5 | API + event pipeline + undo/redo |
| 4–7 | Editor core incl. TranscriptEditor **(milestone: a post-ready Reel without leaving the app)** |
| 6–8 | Product completeness (ranked list) · style prior + profile + E2/E8 · OPE E9 |
| 8–10 | Deploy, studies E6/E11/E12, E7, E9 on real logs **(Review 3)** · A: PPO if ahead |
| 11–12 | Report, demo, buffer |

### MoSCoW (ranked within Should; cut from the bottom)
- **Must:**
  - pipeline + multi-signal features
  - D1, D2 + LinTS + baselines
  - logging (U1), Why? (U2), undo/redo (U3)
  - simulator E1/E3
  - editor with preview, **transcript editor (P1)**, **captions (P2)**, **audio enhancement (P3)**, **reframing (P4)**, **projects/export/SRT (P5)**
  - basic profile
- **Should (in order):**
  1. D4 filler words
  2. D5 caption emphasis
  3. music + ducking
  4. text overlays
  5. review queue + seen tracking (U6, E8)
  6. style prior (U5, E2)
  7. D3 zoom
  8. autopilot/batch (E12)
  9. E11 head-to-head study
  10. OPE (E9)
  11. multi-clip join
  12. nudges + speed
  13. semantic features (U4)
  14. deployment (if not deployed, demo locally)
- **Could:** PPO + pacing comparisons, Neural-Linear, hook D6, LLM scoring, cover image.
- **Won't:** transitions/effects, stickers, B-roll, beat sync, color, multi-cam, auth, generative video.

> Note: E11 is ranked high because it is *the* evidence for the core purpose. If anything Should-level is cut, protect E11 and drop items below it first.

---

## 18. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Whisper slow on CPU hosting | `base.en` int8; ≤ 5 min; caching; progress UI. |
| Whisper drops fillers / timestamps drift | Disfluency prompt; `nonword_voiced_ratio`; 40 ms padding; skip fillers with no clean gap. |
| Filler-word cuts sound choppy | Only suggest REMOVE when gaps ≥ 30 ms on both sides; 8 ms fades; preview to check; the conservative prior removes only um/uh at first. |
| Caption preview ≠ export | Same JSON, same fonts, px at 1080 scale; a parity test renders a frame via ffmpeg and checks line breaks against the preview's line computation (both from `edl/build.py`, not the browser). |
| ffmpeg without libass | Docker image uses a full ffmpeg build; startup check for the `subtitles` filter. |
| Music licensing | CC0/royalty-free only, licenses recorded; user uploads are the user's responsibility (note in UI). |
| Too much product work for 3 people | Ranked Should list; plain-span transcript editor (no rich-text lib); presets instead of free-form styling; no transitions/effects. |
| Not enough real feedback | Simulator is primary evidence; OPE; style prior. |
| Feedback fatigue | Pre-applied suggestions, confidence queue, transcript editing as natural input, autopilot. |
| Implicit/autopilot accepts bias learning | Only seen items count; autopilot exports yield no signal; ablated in E7. |
| E11 unfair comparison | Same raw clip, counterbalanced order, same goal spec, blind raters, participants already familiar with the simple editor. |
| Schema/feature changes | Event log + `context_raw` + `feature_version`; recompute. |
| Scope creep | Sections 1.1 and 2 "not matched/not added" lists are binding. |

---

## 19. Suggested workflow with Sonnet
1. Keep this file at the repo root. Start each session with: *"Read PLAN.md. Implement Phase N only (or Phase 5 item k). Stop when the acceptance criteria pass and show me the test output."*
2. Phase 1: put real clips (incl. a 2-take one) in `data/raw/` so it can run and inspect `units.json` and a rendered MP4.
3. Contracts in `schemas.py` change only when you raise it explicitly. Tell Sonnet never to change them silently.
4. After Phase 2, **look at the E1 plot yourselves**. After Phase 4, **try to make a real Reel end to end yourselves** before adding Phase 5 items. If either fails, fix it first.
5. Could items start only after Phase 8 passes.
