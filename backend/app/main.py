from __future__ import annotations

import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .api.routes import router
from .appctx import AppContext
from .svc.bundle import ApiError
from .svc.projects import recover_interrupted


def _ffmpeg_status() -> dict:
    try:
        out = subprocess.run(
            [config.ffmpeg_path(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
        needed = ["subtitles", "sidechaincompress", "loudnorm", "afftdn", "ebur128"]
        return {"ok": True, "path": config.ffmpeg_path(), "missing_filters": [f for f in needed if f" {f} " not in out]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def create_app(ctx: AppContext | None = None, **ctx_kwargs) -> FastAPI:
    """Build the app. The AppContext (DB, jobs, policies) is created at startup unless one is passed in.

    ctx_kwargs go to AppContext (data_dir, ingest_fn, workers, shipped_priors) - used by the tests.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own = ctx is None
        app.state.ctx = ctx or AppContext(**ctx_kwargs)
        recover_interrupted(app.state.ctx)
        try:
            yield
        finally:
            if own:
                app.state.ctx.close()
                app.state.ctx = None

    app = FastAPI(title="RoughCut", lifespan=lifespan)
    app.state.ctx = ctx

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError) -> JSONResponse:
        return JSONResponse(status_code=e.status, content={"detail": e.detail, **e.extra})

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "ffmpeg": _ffmpeg_status()}

    app.include_router(router)
    return app


app = create_app()
