"""REST routes (PLAN 16.2). Thin: parsing, uploads, status codes; the logic lives in app.svc."""
from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app import config
from app import music as music_lib
from app.api.io import (
    EventIn,
    FinalizeIn,
    FinalizeOut,
    ProfileOut,
    ProjectOut,
    ProjectSummary,
    RenderIn,
    SeenIn,
    StateOut,
    TrackOut,
    UserIn,
    UserOut,
)
from app.appctx import AppContext
from app.models import User
from app.svc import events as ev
from app.svc import projects as pj
from app.svc.bundle import ApiError

router = APIRouter(prefix="/api")
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
_UNSAFE = re.compile(r"[^\w.\- ]")


def safe_filename(raw: str | None, i: int) -> str:
    """A name that is legal on every OS: no path parts, no reserved characters, no trailing dots/spaces."""
    name = _UNSAFE.sub("_", Path(raw or "").name.replace("\\", "/").rsplit("/", 1)[-1]).strip(" .")
    stem, suffix = Path(name).stem[:60], Path(name).suffix[:8]
    return f"{stem}{suffix}" if stem else f"clip{i}{suffix or '.mp4'}"
CHUNK = 1024 * 1024


def get_ctx(request: Request) -> AppContext:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise ApiError(503, "the application is not initialised yet")
    return ctx


def get_session(ctx: AppContext = Depends(get_ctx)) -> Iterator[Session]:
    with ctx.session() as s:
        yield s


# --------------------------------------------------------------------------- users
@router.get("/users", response_model=list[UserOut])
def list_users(session: Session = Depends(get_session)) -> list[User]:
    return list(session.exec(select(User).order_by(User.id)).all())


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserIn, session: Session = Depends(get_session)) -> User:
    return pj.create_user(session, body.name)


@router.get("/users/{user_id}/profile", response_model=ProfileOut)
def user_profile(user_id: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return pj.profile(ctx, session, user_id)


# --------------------------------------------------------------------------- styles / fonts (shared with the preview)
@router.get("/styles")
def styles() -> dict:
    """The caption / text style JSON the export renders from, plus the export presets: the preview reads the same."""
    from app.render.ass import load_styles

    cap, txt = load_styles()
    return {"captions": cap, "texts": txt, "presets": pj.load_presets()}


@router.get("/fonts/{name}")
def font(name: str):
    from app.render.ass import FONTS_DIR

    path = FONTS_DIR / name
    if name not in {p.name for p in FONTS_DIR.glob("*.ttf")} or not path.is_file():
        raise ApiError(404, f"no font {name!r}")
    return FileResponse(path, media_type="font/ttf", headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------------------------------------------------- background music
def _need_user(session: Session, user_id: int) -> None:
    if session.get(User, user_id) is None:
        raise ApiError(404, f"user {user_id} not found")


@router.get("/music", response_model=list[TrackOut])
def list_music(user_id: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    """The bundled tracks and the user's own uploads."""
    _need_user(session, user_id)
    return music_lib.list_tracks(ctx.data_dir, user_id)


@router.get("/music/{track_id}/audio")
def music_audio(track_id: str, user_id: int, ctx: AppContext = Depends(get_ctx)):
    """The audio of a track (for the preview player; Range requests work)."""
    path = music_lib.track_path(ctx.data_dir, user_id, track_id)
    if path is None:
        raise ApiError(404, f"no music track {track_id!r}")
    return FileResponse(path, media_type="audio/mpeg" if path.suffix == ".mp3" else "audio/wav")


@router.post("/users/{user_id}/music", response_model=TrackOut, status_code=201)
def upload_music(user_id: int, file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx),
                 session: Session = Depends(get_session)):
    """Add the user's own track (mp3 / wav, at most 10 MB)."""
    _need_user(session, user_id)
    data = file.file.read(music_lib.MAX_UPLOAD_BYTES + 1)
    try:
        return music_lib.save_upload(ctx.data_dir, user_id, file.filename or "", data)
    except music_lib.MusicError as e:
        raise ApiError(422, str(e)) from None


# --------------------------------------------------------------------------- projects
@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(user_id: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return pj.list_projects(ctx, session, user_id)


def _save_uploads(ctx: AppContext, files: list[UploadFile], order: str | None) -> tuple[list[Path], list[str]]:
    if not files:
        raise ApiError(422, "at least one video file is required")
    if order:
        try:
            idx = json.loads(order)
            ok = (isinstance(idx, list) and all(type(i) is int for i in idx)  # (bool is an int subclass: not an index)
                  and sorted(idx) == list(range(len(files))))
        except ValueError:
            ok = False
        if not ok:
            raise ApiError(422, f"order must be a JSON permutation of 0..{len(files) - 1}")
        files = [files[i] for i in idx]
    tmp_dir = ctx.uploads_dir / f"tmp_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True)
    paths: list[Path] = []
    names: list[str] = []
    total = 0
    try:
        for i, f in enumerate(files):
            name = safe_filename(f.filename, i)
            if Path(name).suffix.lower() not in VIDEO_EXTS:
                raise ApiError(422, f"{name!r} is not a supported video file ({sorted(VIDEO_EXTS)})")
            p = tmp_dir / f"{i:02d}.upload"
            with p.open("wb") as out:
                while chunk := f.file.read(CHUNK):
                    total += len(chunk)
                    if total > config.MAX_TOTAL_BYTES:
                        raise ApiError(413, f"upload is larger than {config.MAX_TOTAL_BYTES // 2**20} MB")
                    out.write(chunk)
            if p.stat().st_size == 0:
                raise ApiError(422, f"{name!r} is empty")
            paths.append(p)
            names.append(name)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return paths, names


@router.post("/projects", response_model=ProjectSummary, status_code=201)
def create_project(files: list[UploadFile] = File(...), user_id: int = Form(...), type: str = Form("other"),
                   target: float | None = Form(None), title: str = Form(""), order: str | None = Form(None),
                   autopilot: bool = Form(False),
                   ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    paths, names = _save_uploads(ctx, files, order)
    try:
        v = pj.create_project(ctx, session, user_id, paths, names, type, target, title, autopilot)
    except BaseException:
        shutil.rmtree(paths[0].parent, ignore_errors=True)
        raise
    shutil.rmtree(paths[0].parent, ignore_errors=True)  # emptied by the move into uploads/v_<id>
    return pj.summary(ctx, v)


@router.get("/projects/{pk}", response_model=ProjectOut)
def get_project(pk: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return pj.get_project(ctx, session, pk)


@router.get("/projects/{pk}/proxy")
def proxy(pk: int, ctx: AppContext = Depends(get_ctx)):
    """The 720p proxy for the browser preview. FileResponse serves HTTP Range requests (seeking works)."""
    path = ctx.video_dir(pk) / "proxy.mp4"
    if not path.exists():
        raise ApiError(404, "the proxy is not ready")
    return FileResponse(path, media_type="video/mp4")


@router.get("/projects/{pk}/thumbnail")
def thumbnail(pk: int, ctx: AppContext = Depends(get_ctx)):
    path = ctx.video_dir(pk) / "thumb.jpg"
    if not path.exists():
        raise ApiError(404, "no thumbnail")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/projects/{pk}/suggest", response_model=StateOut)
def suggest(pk: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return pj.suggest(ctx, session, pk)


@router.get("/projects/{pk}/suggestions/{sid}/explain")
def explain(pk: int, sid: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return pj.explain_suggestion(ctx, session, pk, sid)


@router.post("/projects/{pk}/events", response_model=StateOut)
def post_event(pk: int, body: EventIn, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return ev.handle_event(ctx, session, pk, body)


@router.post("/projects/{pk}/undo", response_model=StateOut)
def undo(pk: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return ev.undo(ctx, session, pk)


@router.post("/projects/{pk}/redo", response_model=StateOut)
def redo(pk: int, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return ev.redo(ctx, session, pk)


@router.post("/projects/{pk}/seen")
def seen(pk: int, body: SeenIn, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)):
    return pj.mark_seen(ctx, session, pk, body.unit_ids)


@router.post("/projects/{pk}/finalize", response_model=FinalizeOut)
def finalize(pk: int, body: FinalizeIn | None = None, ctx: AppContext = Depends(get_ctx),
             session: Session = Depends(get_session)):
    return pj.finalize(ctx, session, pk, (body or FinalizeIn()).seconds_active)


@router.post("/projects/{pk}/render", status_code=202)
def render(pk: int, body: RenderIn | None = None, ctx: AppContext = Depends(get_ctx),
           session: Session = Depends(get_session)):
    return pj.start_render(ctx, session, pk, (body or RenderIn()).preset)


@router.get("/projects/{pk}/download")
def download(pk: int, fmt: str = "mp4", preset: str | None = None, ctx: AppContext = Depends(get_ctx),
             session: Session = Depends(get_session)):
    path, name, info = pj.download_path(ctx, session, pk, fmt, preset)
    media = "video/mp4" if fmt == "mp4" else "application/x-subrip"
    return FileResponse(path, media_type=media, filename=f"v_{pk}_{name}.{fmt}",
                        headers={"X-Render-Stale": "true" if info["stale"] else "false"})


@router.get("/jobs/{job_id}")
def job(job_id: str, ctx: AppContext = Depends(get_ctx)):
    j = ctx.jobs.get(job_id)
    if j is None:
        raise ApiError(404, f"job {job_id} not found")
    return j.public()
