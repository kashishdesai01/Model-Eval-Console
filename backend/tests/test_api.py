import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.db.models import Dataset, EvalRun, Prediction
from app.db.session import get_session
from app.main import create_app
from tests.factories import make_run

pytestmark = pytest.mark.integration


@pytest.fixture
def client(db):
    app = create_app()
    app.dependency_overrides[get_session] = lambda: db
    # No lifespan/model is needed to test registry and contracts.
    yield TestClient(app)


def test_register_validation_conflict_and_enqueue(client, db):
    body = {"name": "new", "hf_model_id": "test/model", "hf_revision": "a" * 40}
    response = client.post("/v1/candidates", json=body)
    assert response.status_code == 201
    assert response.json()["serving_config"]["max_length"] == 128
    assert client.post("/v1/candidates", json=body).status_code == 409
    db.rollback()
    body["hf_revision"] = "main"
    response = client.post("/v1/candidates", json=body)
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    run = make_run(db)
    response = client.post(
        "/v1/runs", json={"candidate_id": str(run.candidate_id), "dataset_id": str(run.dataset_id)}
    )
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_refuses_incomplete_missing_and_mismatched_runs(client, db):
    run = make_run(db)
    body = {"baseline_run_id": str(run.id), "candidate_run_id": str(run.id)}
    assert client.post("/v1/comparisons", json=body).status_code == 422
    body["candidate_run_id"] = str(uuid.uuid4())
    assert client.post("/v1/comparisons", json=body).status_code == 404
    assert client.get(f"/v1/runs/{uuid.uuid4()}").status_code == 404


def test_migration_enforces_registry_immutability(db):
    run = make_run(db)
    for statement in (
        "UPDATE candidates SET name='mutated'",
        "UPDATE datasets SET content_hash='changed'",
        "UPDATE examples SET label=1-label",
    ):
        with pytest.raises(DBAPIError), db.begin_nested():
            db.execute(text(statement))
    assert run.id is not None


def test_persisted_comparison_reproduces_and_refuses_different_dataset(client, db):
    baseline = make_run(db, n=100)
    baseline.status = "succeeded"
    candidate = EvalRun(
        candidate_id=baseline.candidate_id,
        dataset_id=baseline.dataset_id,
        status="succeeded",
        env={},
        code_version="pending",
    )
    db.add(candidate)
    db.flush()
    for idx in range(100):
        db.add(
            Prediction(
                run_id=baseline.id,
                example_idx=idx,
                pred=idx % 2,
                prob_pos=0.9 if idx % 2 else 0.1,
                correct=True,
            )
        )
        pred = 1 - idx % 2 if idx < 20 else idx % 2
        db.add(
            Prediction(
                run_id=candidate.id,
                example_idx=idx,
                pred=pred,
                prob_pos=0.9 if pred else 0.1,
                correct=idx >= 20,
            )
        )
    db.commit()
    body = {"baseline_run_id": str(baseline.id), "candidate_run_id": str(candidate.id)}
    first = client.post("/v1/comparisons", json=body)
    second = client.post("/v1/comparisons", json=body)
    assert first.status_code == second.status_code == 201
    assert first.json()["ci95"] == second.json()["ci95"]
    assert first.json()["verdict"] == "fail"
    assert first.json()["regressions"] == 20
    assert client.get(f"/v1/comparisons/{first.json()['id']}").json() == first.json()
    other = Dataset(
        name="other",
        hf_dataset="test/other",
        hf_config="binary",
        split="test",
        hf_revision="c" * 40,
        content_hash="d" * 64,
        n_examples=1,
    )
    db.add(other)
    db.flush()
    candidate.dataset_id = other.id
    db.commit()
    refused = client.post("/v1/comparisons", json=body)
    assert refused.status_code == 422
    assert refused.json()["type"] == "urn:mec:problem:dataset-mismatch"


def test_invalid_generation_is_persisted_visible_and_not_dropped(client, db):
    baseline = make_run(db, n=100)
    baseline.status = "succeeded"
    candidate = EvalRun(
        candidate_id=baseline.candidate_id, dataset_id=baseline.dataset_id, status="succeeded"
    )
    db.add(candidate)
    db.flush()
    for idx in range(100):
        db.add(
            Prediction(
                run_id=baseline.id,
                example_idx=idx,
                pred=idx % 2,
                prob_pos=0.9 if idx % 2 else 0.1,
                correct=True,
            )
        )
        db.add(
            Prediction(
                run_id=candidate.id,
                example_idx=idx,
                pred=-1 if idx < 10 else idx % 2,
                prob_pos=None,
                raw_output="unclear" if idx < 10 else "positive" if idx % 2 else "negative",
                correct=idx >= 10,
            )
        )
    db.commit()
    response = client.post(
        "/v1/comparisons",
        json={"baseline_run_id": str(baseline.id), "candidate_run_id": str(candidate.id)},
    )
    assert response.status_code == 201
    comparison = response.json()
    assert comparison["n_examples"] == 100 and comparison["regressions"] == 10
    assert comparison["candidate"]["value"] == 0.9
    assert comparison["flips"][0]["candidate_prob"] is None
    assert comparison["flips"][0]["candidate_pred"] == -1
    assert comparison["flips"][0]["candidate_output"] == "unclear"
    assert any("10 invalid outputs" in warning for warning in comparison["warnings"])
    predictions = client.get(f"/v1/runs/{candidate.id}/predictions").json()
    assert predictions[0]["raw_output"] == "unclear" and predictions[0]["prob_pos"] is None
