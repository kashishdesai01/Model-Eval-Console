from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.db.models import EvalRun, Prediction
from app.worker.queue import LostLease, claim_run, complete, heartbeat, write_batch
from app.worker.runner import evaluate
from tests.factories import make_run

pytestmark = pytest.mark.integration


def rows(run_id, start, end):
    return [
        {
            "run_id": run_id,
            "example_idx": i,
            "pred": i % 2,
            "prob_pos": 0.9 if i % 2 else 0.1,
            "correct": True,
        }
        for i in range(start, end)
    ]


def test_skip_locked_claims_and_recovery(db, factory):
    run = make_run(db)
    with factory.begin() as first:
        a = claim_run(first, "worker-a", 60)
        with factory.begin() as second:
            assert claim_run(second, "worker-b", 60) is None
    assert a is not None
    with factory.begin() as session:
        write_batch(session, a, rows(run.id, 0, 10))
    with factory.begin() as session:
        persisted = session.get(EvalRun, run.id)
        persisted.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with factory.begin() as session:
        b = claim_run(session, "worker-b", 60)
    assert b and b.token != a.token and b.attempt == 2
    for operation in (
        lambda s: write_batch(s, a, rows(run.id, 10, 20)),
        lambda s: heartbeat(s, a, 60),
        lambda s: complete(s, a, {}),
    ):
        with pytest.raises(LostLease), factory.begin() as session:
            operation(session)
    with factory.begin() as session:
        write_batch(session, b, rows(run.id, 0, 20))
        complete(session, b, {"accuracy": 1.0})
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 20
        assert session.get(EvalRun, run.id).status == "succeeded"


def test_no_completion_with_missing_or_extra_ids(db, factory):
    run = make_run(db)
    with factory.begin() as session:
        claim = claim_run(session, "worker", 60)
    with pytest.raises(ValueError), factory.begin() as session:
        write_batch(session, claim, rows(run.id, 0, 19))
        complete(session, claim, {})
    with pytest.raises(ValueError), factory.begin() as session:
        write_batch(session, claim, rows(run.id, 1, 21))
        complete(session, claim, {})


def test_retry_budget(db, factory):
    run = make_run(db)
    with factory.begin() as session:
        persisted = session.get(EvalRun, run.id)
        persisted.status, persisted.attempts = "running", 3
        persisted.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with factory.begin() as session:
        assert claim_run(session, "worker", 60) is None
    with factory() as session:
        assert session.get(EvalRun, run.id).status == "failed"


@pytest.mark.parametrize("change", ["code", "environment"])
def test_resume_refuses_provenance_changes(db, factory, monkeypatch, change):
    run = make_run(db)
    with factory.begin() as session:
        claim = claim_run(session, "worker", 60)
        owned = session.get(EvalRun, run.id)
        owned.code_version, owned.env = "v1", {"version": "a"}
        write_batch(session, claim, rows(run.id, 0, 1))
    monkeypatch.setattr(
        "app.worker.runner.code_version", lambda: "v2" if change == "code" else "v1"
    )

    class ChangedInference:
        def environment(self):
            return {"version": "b" if change == "environment" else "a"}

    with pytest.raises(ValueError, match="Resume refused"):
        evaluate(claim, factory, ChangedInference())
