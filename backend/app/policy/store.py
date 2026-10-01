"""Policy-state cache on disk (PLAN 16.1 PolicyState): sufficient statistics as .npz + a small JSON sidecar.

It is only a cache: the event log is the source of truth, so a missing / stale / version-mismatched file is
simply rebuilt with log_replay.recompute_from_log.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.policy.base import Policy


class PolicyStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, user_id: int | str, name: str) -> tuple[Path, Path]:
        stem = f"user{user_id}_{name}"
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    def save(self, user_id: int | str, policy: Policy, n_events: int | None = None) -> None:
        state = policy.state_dict()
        arrays: dict[str, np.ndarray] = {}
        for k, v in state.get("models", {}).items():
            arrays[f"model|{k}|A_d"], arrays[f"model|{k}|b_d"], arrays[f"model|{k}|n"] = v["A_d"], v["b_d"], np.array(v["n"])
        for k, v in state.get("prior", {}).items():
            arrays[f"prior|{k}"] = np.asarray(v)
        npz, meta = self._paths(user_id, policy.name)
        np.savez_compressed(npz, **arrays)
        info = dict(state.get("meta", {}))
        info.update({"policy": policy.name, "version": policy.version, "has": ["prior"] if "prior" in state else [],
                     "n_events": policy.n_updates if n_events is None else n_events})
        meta.write_text(json.dumps(info), encoding="utf-8")

    def meta(self, user_id: int | str, name: str) -> dict | None:
        _, meta = self._paths(user_id, name)
        return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None

    def load_into(self, user_id: int | str, policy: Policy, n_events: int | None = None) -> bool:
        """Restore into `policy`. Returns False (caller should recompute from the log) if absent or stale."""
        npz, _ = self._paths(user_id, policy.name)
        info = self.meta(user_id, policy.name)
        if info is None or not npz.exists():
            return False
        if info.get("version") != policy.version or (n_events is not None and info.get("n_events") != n_events):
            return False
        data = np.load(npz)
        models: dict[str, dict] = {}
        prior: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("prior|"):
                prior[key[len("prior|"):]] = data[key]
                continue
            k, field = key[len("model|"):].rsplit("|", 1)
            models.setdefault(k, {})[field] = data[key]
        for v in models.values():
            v["n"] = int(v["n"])
        state = {"meta": info, "models": models}
        if "prior" in info.get("has", []):
            state["prior"] = prior
        policy.load_state_dict(state)
        return True

    def save_mixture(self, user_id: int | str, mixture) -> None:
        """StylePosterior cache: the log-weights and event count (frozen state follows from n_events)."""
        (self.root / f"user{user_id}_style.json").write_text(
            json.dumps({"names": mixture.names, "logw": mixture.logw.tolist(), "n_events": mixture.n_events}),
            encoding="utf-8")

    def load_mixture(self, user_id: int | str, mixture) -> bool:
        path = self.root / f"user{user_id}_style.json"
        if not path.exists():
            return False
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["names"] != mixture.names:
            return False
        mixture.logw = np.array(d["logw"], dtype=float)
        mixture.n_events = int(d["n_events"])
        mixture._mean = None
        return True
