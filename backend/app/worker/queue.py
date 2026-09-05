import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.metrics import RECLAIMS
from app.db.models import EvalRun, Example, Prediction


class LostLease(Exception):
    pass


@dataclass(frozen=True)
class Claim:
    run_id: uuid.UUID
    token: uuid.UUID
    attempt: int


def db_now(session: Session) -> datetime:
    value = session.scalar(select(func.clock_timestamp()))
    assert isinstance(value, datetime)
    return value


def claim_run(session: Session, worker_id: str, lease_seconds: int) -> Claim | None:
    """Called within a short transaction; never keep a DB lock across inference."""
    while True:
        now = db_now(session)
        run = session.scalar(
            select(EvalRun)
            .where(
                or_(
                    EvalRun.status == "queued",
                    (EvalRun.status == "running") & (EvalRun.lease_expires_at < now),
                )
            )
            .order_by(EvalRun.created_at, EvalRun.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if run is None:
            return None
        if run.attempts >= run.max_attempts:
            run.status, run.finished_at = "failed", now
            run.error = run.error or "Retry budget exhausted after lease expiry"
            run.claim_token, run.lease_expires_at = None, None
            session.flush()
            continue
        if run.status == "running":
            RECLAIMS.inc()
        run.attempts += 1
        run.status, run.worker_id, run.claim_token = "running", worker_id, uuid.uuid4()
        run.started_at = run.started_at or now
        run.lease_expires_at = now + timedelta(seconds=lease_seconds)
        run.error = None
        session.flush()
        return Claim(run.id, run.claim_token, run.attempts)


def owned_run(session: Session, claim: Claim) -> EvalRun:
    run = session.scalar(
        select(EvalRun)
        .where(
            EvalRun.id == claim.run_id,
            EvalRun.status == "running",
            EvalRun.claim_token == claim.token,
        )
        .with_for_update()
    )
    if run is None or run.lease_expires_at is None or run.lease_expires_at <= db_now(session):
        raise LostLease("Claim expired or was replaced")
    return run


def heartbeat(session: Session, claim: Claim, lease_seconds: int) -> None:
    run = owned_run(session, claim)
    run.lease_expires_at = db_now(session) + timedelta(seconds=lease_seconds)


def write_batch(session: Session, claim: Claim, rows: list[dict[str, Any]]) -> None:
    owned_run(session, claim)
    if any(row["run_id"] != claim.run_id for row in rows):
        raise ValueError("Batch contains another run")
    if rows:
        session.execute(
            insert(Prediction)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=[Prediction.run_id, Prediction.example_idx],
            )
        )


def complete(session: Session, claim: Claim, metrics: dict[str, Any]) -> None:
    run = owned_run(session, claim)
    expected = set(session.scalars(select(Example.idx).where(Example.dataset_id == run.dataset_id)))
    actual = set(session.scalars(select(Prediction.example_idx).where(Prediction.run_id == run.id)))
    if not expected or expected != actual:
        raise ValueError("Cannot complete: prediction IDs do not exactly match the dataset")
    run.status, run.metrics, run.finished_at = "succeeded", metrics, db_now(session)
    run.claim_token, run.lease_expires_at = None, None


def fail(session: Session, claim: Claim, error: str) -> None:
    run = owned_run(session, claim)
    run.error = error[:2000]
    run.status = "failed" if run.attempts >= run.max_attempts else "queued"
    if run.status == "failed":
        run.finished_at = db_now(session)
    run.claim_token, run.lease_expires_at = None, None
