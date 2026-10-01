"""Build the shipped priors in backend/assets/priors from the persona simulator (PLAN 7.2, 15).

generic.json    pooled generic prior over all personas (theta^(0))
prototypes.json one ridge-fit style prototype per persona (theta^(1..K))

    python backend/scripts/build_prototypes.py   (from the repo root, with the backend venv active)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.experiments.common import generic_prior, prototype
from sim.personas import BASE_PERSONAS

OUT = Path(__file__).resolve().parents[1] / "assets" / "priors"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generic.json").write_text(json.dumps(generic_prior(()).to_dict()), encoding="utf-8")
    (OUT / "prototypes.json").write_text(
        json.dumps({n: prototype(n).to_dict() for n in BASE_PERSONAS}), encoding="utf-8")
    print("wrote", OUT / "generic.json", "and", OUT / "prototypes.json")


if __name__ == "__main__":
    main()
