"""Re-run the named simulator experiments from the current code; summaries and raw plots go to paper/data/e.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\rerun_e.py [e2 e3 e4 e5 e8 e9 e10]
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "e"
ALL = ("e2", "e3", "e4", "e5", "e8", "e9", "e10")

if __name__ == "__main__":
    for name in sys.argv[1:] or ALL:
        t0 = time.time()
        m = importlib.import_module(f"sim.experiments.{name}")
        m.report(m.run(workers=6), OUT)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
