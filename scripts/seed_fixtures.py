"""Synthetic stored predictions for UI/CLI tests; NEVER presented as measured inference."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.api.schemas import ComparisonCreate, ServingConfig  # noqa: E402
from app.db.models import Candidate, Dataset, EvalRun, Example, Prediction  # noqa: E402
from app.db.session import session_factory  # noqa: E402
from app.services.comparisons import create_comparison  # noqa: E402
from app.services.datasets import content_hash, slice_tags  # noqa: E402


def seed() -> dict:
    with session_factory()() as session:
        existing = session.scalar(select(Dataset).where(Dataset.name == "Synthetic CI fixture"))
        if existing:
            raise RuntimeError("Fixture already exists. Use a fresh fixture database.")
        text = (
            "The opening was dull and slow, with weak acting and a predictable plot, "
            "but the final scenes were beautifully written "
            "and made this a surprisingly wonderful movie."
        )
        rows = [
            (i, text if i % 2 else "A poor, boring movie with no redeeming qualities.", i % 2)
            for i in range(200)
        ]
        dataset = Dataset(
            name="Synthetic CI fixture",
            hf_dataset="fixture/sentiment",
            hf_config="binary",
            hf_revision="f" * 40,
            split="fixture",
            n_examples=len(rows),
            content_hash=content_hash(rows),
        )
        session.add(dataset)
        session.flush()
        session.add_all(
            [
                Example(dataset_id=dataset.id, idx=i, text=t, label=label, slices=slice_tags(t))
                for i, t, label in rows
            ]
        )
        session.commit()
        runs = {}
        for name in (
            "fixture-baseline",
            "fixture-noop",
            "fixture-borderline",
            "fixture-truncation",
        ):
            candidate = Candidate(
                name=name,
                hf_model_id="fixture/not-a-model",
                hf_revision="f" * 40,
                serving_config=ServingConfig(
                    max_length=16 if name.endswith("truncation") else 128
                ).model_dump(),
            )
            session.add(candidate)
            session.flush()
            run = EvalRun(
                candidate_id=candidate.id,
                dataset_id=dataset.id,
                status="succeeded",
                attempts=1,
                env={"evidence": "fixture"},
                code_version="fixture-v1",
                finished_at=datetime.now(UTC),
                metrics={},
            )
            session.add(run)
            session.flush()
            correct_count = 0
            for i, _, label in rows:
                pred = label
                if i < 10:
                    pred = 1 - label
                if name.endswith("truncation") and i >= 10 and i % 2:
                    pred = 0
                if name.endswith("borderline"):
                    if 10 <= i < 20:
                        pred = 1 - label
                    if i < 8:
                        pred = label
                correct_count += pred == label
                session.add(
                    Prediction(
                        run_id=run.id,
                        example_idx=i,
                        pred=pred,
                        prob_pos=0.93 if pred else 0.07,
                        correct=pred == label,
                    )
                )
            run.metrics = {"accuracy": correct_count / len(rows)}
            runs[name] = run
            session.commit()
        comparisons = {}
        for name in ("fixture-noop", "fixture-borderline", "fixture-truncation"):
            comparison = create_comparison(
                session,
                ComparisonCreate(
                    baseline_run_id=runs["fixture-baseline"].id,
                    candidate_run_id=runs[name].id,
                ),
            )
            comparisons[name] = {"id": str(comparison.id), "verdict": comparison.verdict}
        return {
            "evidence": "fixture",
            "runs": {name: str(run.id) for name, run in runs.items()},
            "comparisons": comparisons,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/fixture-results.json")
    args = parser.parse_args()
    result = seed()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
