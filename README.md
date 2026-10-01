# RoughCut

Human-in-the-loop RL for personalized short-form video editing. See `PLAN.md` for the full design and `CLAUDE.md` for dev commands.

## Test clips are placeholders
`scripts/make_test_clips.py` generates synthetic clips (Windows TTS voice over a plain background, with pauses, fillers and a retake; two short "takes" for multi-clip join) into `data/raw/`. **Real face-to-camera clips are needed later** for face tracking, the dataset, and the user studies.
