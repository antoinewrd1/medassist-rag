"""Query logging via SQLAlchemy.

SQLite locally, Postgres under docker-compose -- one URL switch, no code change.

PRIVACY POSTURE: this prototype logs the symptom text because the whole point
of the log is to review what the system said and why. That is defensible for a
demo with synthetic input and indefensible for real patients: symptom text is
PHI. A production deployment needs a retention policy, encryption at rest, an
access log, and a HIPAA-covered hosting agreement. `store_text=False` disables
symptom persistence entirely and is the setting a real deployment would start
from while those controls are built.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)
    triage: Mapped[str] = mapped_column(String(20))
    matched_rules: Mapped[str] = mapped_column(String(500), default="")
    citations: Mapped[str] = mapped_column(String(500), default="")
    llm_invoked: Mapped[int] = mapped_column(Integer, default=0)
    grounded: Mapped[int] = mapped_column(Integer, default=0)
    backend: Mapped[str] = mapped_column(String(100), default="")


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().db_url, future=True)
        Base.metadata.create_all(_engine)
    return _engine


def log_query(response, symptoms: str, store_text: bool = True) -> int | None:
    """Persist one interaction. Never raises -- logging must not break serving."""
    try:
        with Session(get_engine()) as session:
            row = QueryLog(
                created_at=datetime.now(timezone.utc),
                symptoms=symptoms if store_text else None,
                triage=response.triage.value,
                matched_rules=json.dumps(response.matched_safety_rules)[:500],
                citations=json.dumps([c.doc_id for c in response.citations])[:500],
                llm_invoked=int(response.llm_invoked),
                grounded=int(response.grounded),
                backend=response.backend[:100],
            )
            session.add(row)
            session.commit()
            return row.id
    except Exception:  # noqa: BLE001 -- a logging failure must not fail the request
        return None
