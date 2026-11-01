"""Re-run the named simulator experiments from the current code; summaries and raw plots go to paper/data/e.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\rerun_e.py [e1 e2 e3 e4 e5 e8 e9 e10 e11]
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "e"
ALL = ("e1", "e2", "e3", "e4", "e5", "e8", "e9", "e10", "e11")

if __name__ == "__main__":
    for name in sys.argv[1:] or ALL:
        t0 = time.time()
        m = importlib.import_module(f"sim.experiments.{name}")
        if name == "e11":  # its settings were tuned on seeds 1-6: report on seeds never used for that
            raw = m.run(seeds=20, workers=6, seed_base=100)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "e11_raw.json").write_text(json.dumps(raw), encoding="utf-8")
        else:
            raw = m.run(workers=6)
        m.report(raw, OUT)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
