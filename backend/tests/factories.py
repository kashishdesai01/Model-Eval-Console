from app.api.schemas import ServingConfig
from app.db.models import Candidate, Dataset, EvalRun, Example
from app.services.datasets import content_hash


def make_run(session, n=20):
    candidate = Candidate(
        name="test-baseline",
        hf_model_id="test/model",
        hf_revision="a" * 40,
        serving_config=ServingConfig().model_dump(),
    )
    rows = [(i, f"text {i}", i % 2) for i in range(n)]
    dataset = Dataset(
        name="test-data",
        hf_dataset="test/data",
        hf_config="binary",
        split="test",
        hf_revision="b" * 40,
        content_hash=content_hash(rows),
        n_examples=n,
    )
    session.add_all([candidate, dataset])
    session.flush()
    session.add_all(
        [
            Example(dataset_id=dataset.id, idx=idx, text=text, label=label, slices=[])
            for idx, text, label in rows
        ]
    )
    run = EvalRun(candidate_id=candidate.id, dataset_id=dataset.id)
    session.add(run)
    session.commit()
    return run
