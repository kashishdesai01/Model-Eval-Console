import multiprocessing
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import EvalRun, Prediction
from app.inference.service import BatchResult
from app.worker.queue import claim_run
from app.worker.runner import evaluate
from tests.factories import make_run

pytestmark = pytest.mark.integration


class SlowInference:
    def environment(self):
        return {"evidence": "fixture", "version": "slow-test-v1"}

    def predict(self, _model_id, _revision, _config, texts):
        time.sleep(0.1)
        labels = [int(text.split()[-1]) % 2 for text in texts]
        return BatchResult(labels, [0.9 if label else 0.1 for label in labels], 100, False)

    def benchmark(self, *_args):
        return {"latency_p95_ms": 100}


def execute(url, claim):
    engine = create_engine(url, connect_args={"options": "-csearch_path=mec_test"})
    evaluate(claim, sessionmaker(engine, expire_on_commit=False), SlowInference())
    engine.dispose()


def test_actual_process_kill_and_resume(db, factory):
    run = make_run(db, n=80)
    with factory.begin() as session:
        claim = claim_run(session, "process-a", 60)
    url = factory.kw["bind"].url.render_as_string(hide_password=False)
    process = multiprocessing.get_context("spawn").Process(target=execute, args=(url, claim))
    process.start()
    try:
        deadline = time.monotonic() + 30
        count = 0
        while time.monotonic() < deadline:
            with factory() as session:
                count = session.scalar(select(func.count()).select_from(Prediction))
            if count:
                break
            time.sleep(0.05)
        assert 0 < count < 80
        process.kill()
        process.join(timeout=5)
        with factory.begin() as session:
            session.get(EvalRun, run.id).lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        with factory.begin() as session:
            resumed = claim_run(session, "process-b", 60)
        execute(url, resumed)
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(Prediction)) == 80
            assert session.get(EvalRun, run.id).status == "succeeded"
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
