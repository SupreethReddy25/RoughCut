"""Generate synthetic placeholder clips in data/raw/ using Windows TTS + ffmpeg.

clip_01_main.mp4   ~60s: long pauses, fillers (um/uh/like/you know), one retake
clip_02_take_a.mp4 ~20s  } two takes meant to be joined as one video
clip_03_take_b.mp4 ~20s  }
clip_04_messy.mp4  ~60s: English stress test for the analysis: hesitations, um/uh, a stutter, hedges,
                   a three-attempt restart chain, and a line said twice where the FIRST delivery is the clean one

Run with --only clip_04_messy to (re)build a single clip.

Run:  backend\\.venv\\Scripts\\python.exe scripts\\make_test_clips.py
These are PLACEHOLDERS (synthetic voice, no face). Real face clips are needed for face tracking and the dataset.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
from app import config

RAW = config.RAW_DIR
RATE = 22050  # TTS output sample rate (set via format below)

# ("say", text) or ("pause", seconds)
MAIN = [
    ("say", "Hey everyone, today I want to talk about, um, three mistakes beginners make when editing videos."),
    ("pause", 1.4),
    ("say", "The first mistake is, uh, keeping every single pause."),
    ("pause", 0.8),
    ("say", "It is, like, really important to cut the dead air."),
    ("pause", 1.9),
    ("say", "So the second thing is, you know, the pacing of your cuts."),
    ("pause", 0.6),
    ("say", "Fast cuts keep viewers watching for the"),   # abandoned sentence ...
    ("pause", 1.1),
    ("say", "Fast cuts keep viewers watching for much longer."),  # ... retake
    ("pause", 0.9),
    ("say", "Third, um, always add captions because most people watch without sound."),
    ("pause", 1.6),
    ("say", "I was, uh, surprised how much captions changed my watch time."),
    ("pause", 0.5),
    ("say", "In total I saw, like, forty percent more views."),
    ("pause", 1.2),
    ("say", "So that is it. Try these three things and, you know, let me know how it goes."),
    ("pause", 0.7),
    ("say", "Thanks for watching, and see you in the next one."),
]
TAKE_A = [
    ("say", "Okay, so this is take one of my short intro."),
    ("pause", 0.9),
    ("say", "Today we are looking at, um, how to record better audio."),
    ("pause", 1.0),
    ("say", "First, get close to the microphone."),
]
TAKE_B = [
    ("say", "And this is take two, which continues the same video."),
    ("pause", 0.7),
    ("say", "Second, uh, record in a quiet room."),
    ("pause", 1.3),
    ("say", "That is all, thanks for watching."),
]

MESSY = [
    ("pause", 1.2),
    ("say", "Okay, so today I am going to show you how I light my videos."),
    ("pause", 0.9),
    ("say", "The first thing you need is, um, a big soft light."),
    ("pause", 0.35),
    ("say", "Put it, uh, slightly to the side of your face."),
    ("pause", 1.6),
    ("say", "The second tip is to use"),            # attempt 1 (abandoned)
    ("pause", 0.7),
    ("say", "the second tip,"),                     # attempt 2 (abandoned)
    ("pause", 0.5),
    ("say", "The second tip is to use a darker background."),  # final take
    ("pause", 0.8),
    ("say", "I I think this is kind of the most underrated trick."),  # stutter + hedge
    ("pause", 2.4),
    ("say", "Lighting matters more than the camera you use."),  # clean take ...
    ("pause", 0.9),
    ("say", "Lighting matters, um, more than the, uh, camera you use."),  # ... worse repeat
    ("pause", 0.6),
    ("say", "Well, that is basically it for today."),
    ("pause", 0.45),
    ("say", "Let me know what you, like, want to see next."),
    ("pause", 1.5),
]


def tts_to_wavs(items: list[tuple[str, str]]) -> None:
    """items: [(text, wav_path)] -> synthesize with System.Speech in ONE powershell call."""
    spec = Path(tempfile.mkdtemp()) / "tts.json"
    spec.write_text(json.dumps([{"text": t, "out": str(o)} for t, o in items]), encoding="utf-8")
    ps = f"""
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Path '{spec}' | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($i in $items) {{ $s.SetOutputToWaveFile($i.out, $fmt); $s.Speak($i.text); $s.SetOutputToNull() }}
"""
    import os
    exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    subprocess.run([exe, "-NoProfile", "-Command", ps], check=True)


def build_audio(script: list[tuple[str, object]], out_wav: Path, work: Path) -> float:
    say = [(str(v), work / f"{out_wav.stem}_{i}.wav") for i, (k, v) in enumerate(script) if k == "say"]
    tts_to_wavs(say)
    frames = bytearray()
    lead = bytes(int(RATE * 0.4) * 2)  # 0.4 s leading silence
    frames += lead
    si = 0
    for k, v in script:
        if k == "pause":
            frames += bytes(int(RATE * float(v)) * 2)
        else:
            with wave.open(str(say[si][1]), "rb") as w:
                assert w.getframerate() == RATE and w.getnchannels() == 1
                frames += w.readframes(w.getnframes())
            si += 1
            frames += bytes(int(RATE * 0.12) * 2)  # tiny gap after each utterance
    frames += bytes(int(RATE * 0.5) * 2)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return len(frames) / 2 / RATE


def make_video(wav: Path, dur: float, title: str, out: Path) -> None:
    font = "C\\:/Windows/Fonts/arial.ttf"
    vf = (
        f"drawtext=fontfile='{font}':text='{title}':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2,"
        f"drawtext=fontfile='{font}':text='placeholder test clip':fontcolor=0xcccccc:fontsize=36:x=(w-text_w)/2:y=h/2+80"
    )
    subprocess.run(
        [
            config.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x203040:s=1920x1080:r=30:d={dur:.2f}",
            "-i", str(wav), "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-ar", "48000", "-shortest", str(out),
        ],
        check=True,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="build just this clip (e.g. clip_04_messy)")
    only = ap.parse_args().only
    RAW.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    for name, script, title in [
        ("clip_01_main", MAIN, "Clip 1 (main)"),
        ("clip_02_take_a", TAKE_A, "Clip 2 (take A)"),
        ("clip_03_take_b", TAKE_B, "Clip 3 (take B)"),
        ("clip_04_messy", MESSY, "Clip 4 (messy English)"),
    ]:
        if only and name != only:
            continue
        wav = work / f"{name}.wav"
        dur = build_audio(script, wav, work)
        out = RAW / f"{name}.mp4"
        make_video(wav, dur, title, out)
        print(f"{out}  {dur:.1f}s")


if __name__ == "__main__":
    main()
