"""Database layer (SQLAlchemy 2.0).

Defaults to a local SQLite file so the API runs with zero setup.
For PostgreSQL (as in the architecture diagram) set, for example:
    DATABASE_URL=postgresql+psycopg://finguard:finguard@localhost:5432/finguard
"""
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from src.finguard.config import ROOT

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'finguard.db'}")
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Investigation(Base):
    """One agent investigation of one transaction, plus the human review of it."""

    __tablename__ = "investigations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trans_num: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # what the ML layer said at the time
    ml_risk_score: Mapped[float] = mapped_column(Float)
    anomaly_percentile: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)

    # what the agents decided
    decision: Mapped[str] = mapped_column(String(16), index=True)  # allow | review | escalate
    risk_level: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[str] = mapped_column(String(16))
    key_reasons: Mapped[list] = mapped_column(JSON, default=list)
    recommended_action: Mapped[str] = mapped_column(Text, default="")
    requires_human_review: Mapped[bool] = mapped_column(Boolean, default=True)

    # full agent reports (the evidence trail)
    transaction_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    account_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    investigation_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="agents")  # "agents" or "mock"

    # human-in-the-loop
    status: Mapped[str] = mapped_column(String(24), default="pending_review", index=True)
    human_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    audit_events: Mapped[list["AuditEvent"]] = relationship(
        back_populates="investigation", order_by="AuditEvent.id"
    )


class AuditEvent(Base):
    """Append-only log: who did what, and when."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    investigation_id: Mapped[int] = mapped_column(ForeignKey("investigations.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    event_type: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    investigation: Mapped[Investigation] = relationship(back_populates="audit_events")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
