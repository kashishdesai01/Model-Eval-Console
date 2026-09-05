import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.schemas import ServingConfig
from app.core.config import get_settings
from app.db.models import Candidate, Example, Prediction
from app.inference.service import InferenceService
from app.worker.queue import Claim, complete, owned_run, write_batch

logger = logging.getLogger(__name__)


def code_version() -> str:
    configured = get_settings().code_version
    if configured != "development":
        return configured
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return f"source-sha256:{digest.hexdigest()}"


def quality_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    truth, pred = np.asarray(labels), np.asarray(predictions)
    f1s = []
    for label in (0, 1):
        tp = int(((truth == label) & (pred == label)).sum())
        denominator = int((truth == label).sum() + (pred == label).sum())
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": float((truth == pred).mean()),
        "macro_f1": float(np.mean(f1s)),
        "invalid_outputs": int((pred == -1).sum()),
        "invalid_output_rate": float((pred == -1).mean()),
    }


def evaluate(claim: Claim, factory: sessionmaker[Session], inference: InferenceService) -> None:
    with factory.begin() as session:
        run = owned_run(session, claim)
        candidate = session.get(Candidate, run.candidate_id)
        assert candidate is not None
        config = ServingConfig.model_validate(candidate.serving_config)
        model_id, revision = candidate.hf_model_id, candidate.hf_revision
        examples = list(
            session.scalars(
                select(Example)
                .where(
                    Example.dataset_id == run.dataset_id,
                )
                .order_by(Example.idx)
            )
        )
        existing = set(
            session.scalars(
                select(Prediction.example_idx).where(
                    Prediction.run_id == run.id,
                )
            )
        )
        environment = inference.environment()
        version = code_version()
        if existing and (run.env != environment or run.code_version != version):
            raise ValueError("Resume refused: evaluator code or environment changed")
        run.env, run.code_version = environment, version
    texts = [example.text for example in examples]
    probe = texts[: 10 if config.inference_kind == "generative" else 100]
    first = inference.predict(model_id, revision, config, probe)
    second = inference.predict(model_id, revision, config, probe)
    if first.labels != second.labels:
        raise ValueError("Stability probe changed predicted labels across identical calls")
    pending = [example for example in examples if example.idx not in existing]
    for start in range(0, len(pending), config.batch_size):
        batch = pending[start : start + config.batch_size]
        result = inference.predict(model_id, revision, config, [example.text for example in batch])
        rows: list[dict[str, Any]] = [
            {
                "run_id": claim.run_id,
                "example_idx": example.idx,
                "pred": pred,
                "prob_pos": probability,
                "correct": pred == example.label,
                "raw_output": output,
            }
            for example, pred, probability, output in zip(
                batch,
                result.labels,
                result.probabilities,
                result.raw_outputs or [None] * len(batch),
                strict=True,
            )
        ]
        with factory.begin() as session:
            write_batch(session, claim, rows)
        logger.debug(
            "Prediction batch persisted", extra={"run_id": claim.run_id, "attempt": claim.attempt}
        )
    latency = inference.benchmark(model_id, revision, config, texts)
    with factory.begin() as session:
        predictions = list(
            session.scalars(
                select(Prediction)
                .where(
                    Prediction.run_id == claim.run_id,
                )
                .order_by(Prediction.example_idx)
            )
        )
        metrics = quality_metrics(
            [example.label for example in examples], [p.pred for p in predictions]
        )
        complete(session, claim, {**metrics, **latency})
