"""Background jobs (PLAN 5 `jobs`): a small ThreadPoolExecutor with a registry the API can poll."""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | failed
    error: str | None = None
    result: object = None
    meta: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "error": self.error, "meta": self.meta}


class JobRunner:
    def __init__(self, workers: int = 2, ingest_workers: int = 1):
        """Analysis (Whisper, minutes) runs on its own pool, one at a time, so a queue of uploads can never starve
        the short jobs (renders) of worker threads."""
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="roughcut-job")
        self._ingest_pool = ThreadPoolExecutor(max_workers=ingest_workers, thread_name_prefix="roughcut-ingest")
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[], object], kind: str = "job", meta: dict | None = None) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, meta=dict(meta or {}))
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn()
                job.status = "done"
            except Exception as e:  # noqa: BLE001 - a job must never take the worker down
                log.error("job %s (%s) failed:\n%s", job.id, kind, traceback.format_exc())
                job.error = f"{type(e).__name__}: {e}"
                job.status = "failed"
            finally:
                job.finished = time.time()

        fut = (self._ingest_pool if kind == "ingest" else self._pool).submit(run)
        with self._lock:
            self._futures[job.id] = fut
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        with self._lock:
            fut = self._futures[job_id]
        fut.result(timeout)
        return self._jobs[job_id]

    def wait_idle(self, timeout: float | None = None) -> None:
        with self._lock:
            futs = list(self._futures.values())
        for f in futs:
            f.result(timeout)

    def shutdown(self) -> None:
        self._ingest_pool.shutdown(wait=True, cancel_futures=True)
        self._pool.shutdown(wait=True, cancel_futures=True)


class LockMap:
    """Re-entrant locks by key (one per video / user), created on demand."""

    def __init__(self) -> None:
        self._locks: dict[object, threading.RLock] = {}
        self._guard = threading.Lock()

    def __call__(self, key: object) -> threading.RLock:
        with self._guard:
            lk = self._locks.get(key)
            if lk is None:
                lk = self._locks[key] = threading.RLock()
            return lk
