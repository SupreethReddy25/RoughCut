"""Face track (PLAN 8.4, 12): OpenCV YuNet at 5 fps on the proxy -> per-sample face box, per-span stats.

Degrades gracefully: no face / no model / no OpenCV -> empty track -> face_present 0, crop centre (0.5, 0.5).
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app import config

log = logging.getLogger(__name__)

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_FILE
TRACK_FPS = 5.0
SCORE_THRESHOLD = 0.7
DETECT_WIDTH = 640  # frames are downscaled to this width for detection


@dataclass
class FaceSample:
    t: float
    cx: float  # box centre, 0..1 of frame width/height
    cy: float
    w: float  # box size, 0..1 of frame width/height
    h: float
    score: float


def model_path() -> Path:
    p = config.ASSETS_DIR / MODEL_FILE
    if not p.exists():
        log.info("downloading YuNet model from %s", MODEL_URL)
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


def track_faces(video_path, fps: float = TRACK_FPS) -> tuple[list[FaceSample], float]:
    """Returns (samples with a detected face, number of frames sampled). Largest face wins per frame."""
    try:
        import cv2

        detector_cls = cv2.FaceDetectorYN
        mp = model_path()
    except Exception as e:  # noqa: BLE001 - degrade, never fail the pipeline over vision
        log.warning("face tracking unavailable (%s); continuing without faces", e)
        return [], 0
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / fps))
    det = None
    samples: list[FaceSample] = []
    n_sampled = 0
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                if w > DETECT_WIDTH:
                    frame = cv2.resize(frame, (DETECT_WIDTH, round(h * DETECT_WIDTH / w)))
                    h, w = frame.shape[:2]
                if det is None:
                    det = detector_cls.create(str(mp), "", (w, h), SCORE_THRESHOLD, 0.3, 5000)
                det.setInputSize((w, h))
                _, faces = det.detect(frame)
                n_sampled += 1
                if faces is not None and len(faces):
                    f = max(faces, key=lambda r: r[2] * r[3])  # x, y, w, h, landmarks..., score
                    x, y, bw, bh = (float(v) for v in f[:4])
                    samples.append(FaceSample(round(i / src_fps, 3), round((x + bw / 2) / w, 4),
                                              round((y + bh / 2) / h, 4), round(bw / w, 4), round(bh / h, 4),
                                              round(float(f[-1]), 3)))
        i += 1
    cap.release()
    return samples, n_sampled


@dataclass
class FaceStats:
    present: float  # fraction of in-span samples with a face (0 when there is no track)
    cx: float = 0.5
    cy: float = 0.5
    size: float = 0.0  # median box width, fraction of frame width


def face_stats(samples: list[FaceSample], t0: float, t1: float, sample_dt: float = 1 / TRACK_FPS,
               has_track: bool = True) -> FaceStats:
    """Stats for [t0, t1]. Samples are only the frames with a detection, so presence = hits / expected."""
    if not has_track or not samples:
        return FaceStats(0.0)
    hits = [s for s in samples if t0 - 1e-9 <= s.t <= t1 + 1e-9]
    expected = max(1.0, (t1 - t0) / sample_dt)
    if not hits:
        return FaceStats(0.0)
    return FaceStats(
        present=float(min(1.0, len(hits) / expected)),
        cx=float(np.median([s.cx for s in hits])),
        cy=float(np.median([s.cy for s in hits])),
        size=float(np.median([s.w for s in hits])),
    )


def track_to_json(samples: list[FaceSample], n_sampled: float, fps: float = TRACK_FPS) -> dict:
    return {"fps": fps, "frames_sampled": n_sampled, "faces_found": len(samples),
            "samples": [asdict(s) for s in samples]}
