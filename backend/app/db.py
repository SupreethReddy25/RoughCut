"""SQLite engine + schema creation."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import SQLModel, create_engine

from app import models  # noqa: F401  (registers the tables on SQLModel.metadata)


def make_engine(db_path: str | Path) -> Engine:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    SQLModel.metadata.create_all(engine)
    return engine
