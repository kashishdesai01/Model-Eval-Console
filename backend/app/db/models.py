import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    hf_model_id: Mapped[str] = mapped_column(String(200))
    hf_revision: Mapped[str] = mapped_column(String(40))
    serving_config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("hf_dataset", "hf_config", "split", "hf_revision"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    hf_dataset: Mapped[str] = mapped_column(String(200))
    hf_config: Mapped[str] = mapped_column(String(100))
    split: Mapped[str] = mapped_column(String(100))
    hf_revision: Mapped[str] = mapped_column(String(40))
    content_hash: Mapped[str] = mapped_column(String(64))
    n_examples: Mapped[int] = mapped_column(Integer)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Example(Base):
    __tablename__ = "examples"
    __table_args__ = (CheckConstraint("label IN (0, 1)"),)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[int] = mapped_column(SmallInteger)
    slices: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','succeeded','failed')"),
        CheckConstraint("attempts >= 0 AND max_attempts > 0"),
        Index("ix_eval_runs_status_lease", "status", "lease_expires_at"),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id"))
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    worker_id: Mapped[str | None] = mapped_column(Text)
    claim_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    code_version: Mapped[str] = mapped_column(Text, default="pending")
    env: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("pred IN (-1,0,1)", name="ck_prediction_label"),
        CheckConstraint("prob_pos >= 0 AND prob_pos <= 1"),
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    example_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    pred: Mapped[int] = mapped_column(SmallInteger)
    prob_pos: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_output: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool] = mapped_column(Boolean)


class Comparison(Base):
    __tablename__ = "comparisons"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    baseline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("eval_runs.id"))
    candidate_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("eval_runs.id"))
    metric: Mapped[str] = mapped_column(Text, default="accuracy")
    margin: Mapped[float] = mapped_column(Float)
    delta: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)
    n_resamples: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    mcnemar_p: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
