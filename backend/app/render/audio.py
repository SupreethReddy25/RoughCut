"""Audio enhancement chain + loudness measurement (PLAN 8.3).

Chain on export: highpass 80 Hz -> afftdn (light; skipped when the source noise floor is already low)
-> two-pass loudnorm to -14 LUFS / -1.5 dBTP / LRA 11. The 8 ms fades at every cut are added per clip in graph.py.
"""
from __future__ import annotations

import json
import re
import subprocess

from app import config
from app.pipeline.pauses import energy_from_wav

TARGET_I = -14.0
TARGET_TP = -1.5
TARGET_LRA = 11.0
DENOISE_SKIP_BELOW_DB = -55.0  # source noise floor (p10 of 20 ms RMS) quieter than this -> skip afftdn


def source_noise_floor(wav_path) -> float | None:
    try:
        return energy_from_wav(wav_path).floor_db
    except (OSError, ValueError):
        return None


def should_denoise(floor_db: float | None) -> bool:
    return floor_db is None or floor_db > DENOISE_SKIP_BELOW_DB


def enhance_chain(denoise: bool) -> list[str]:
    return ["highpass=f=80"] + (["afftdn=nf=-25"] if denoise else [])


def loudnorm_pass1(target: float = TARGET_I) -> str:
    return f"loudnorm=I={target}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json"


def loudnorm_pass2(m: dict, target: float = TARGET_I) -> str:
    """Second pass: apply the measured stats so the gain is linear (no pumping)."""
    return (f"loudnorm=I={target}:TP={TARGET_TP}:LRA={TARGET_LRA}:measured_I={m['input_i']}:"
            f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true:print_format=summary")


def parse_loudnorm_json(stderr: str) -> dict:
    """The JSON block loudnorm prints at the end of pass 1."""
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.DOTALL)
    if not blocks:
        raise RuntimeError("loudnorm pass 1 printed no stats:\n" + stderr[-1500:])
    return json.loads(blocks[-1])


def measure_loudness(path) -> dict:
    """Integrated loudness / LRA / true peak of a file via ffmpeg ebur128."""
    p = subprocess.run([config.ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(path), "-vn",
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    tail = p.stderr[p.stderr.rfind("Summary:"):] if "Summary:" in p.stderr else p.stderr
    def grab(pat: str) -> float:
        m = re.findall(pat, tail)
        return float(m[-1]) if m else float("nan")
    return {"I": grab(r"I:\s+(-?[\d.]+) LUFS"), "LRA": grab(r"LRA:\s+(-?[\d.]+) LU"),
            "TP": grab(r"Peak:\s+(-?[\d.]+) dBFS")}
