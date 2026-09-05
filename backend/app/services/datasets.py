import hashlib
import json
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import DatasetCreate
from app.db.models import Dataset, Example


def slice_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    if len(text.split()) > 25:
        tags.append("len_long")
    if re.search(r"\b(not|no|never)\b|n['’]t\b", lowered):
        tags.append("has_negation")
    if re.search(r"\b(but|however)\b", lowered):
        tags.append("has_contrast")
    return tags


def content_hash(rows: list[tuple[int, str, int]]) -> str:
    digest = hashlib.sha256()
    for idx, text, label in rows:
        digest.update(
            json.dumps([idx, text, label], ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def imdb_sample(source: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    """Fixed 200 per label, ranked by SHA256(seed:source-index), without RNG drift."""
    selected = []
    for label in (0, 1):
        pool = [row for row in source if row[2] == label]
        if len(pool) < 200:
            raise ValueError("IMDb test source must contain at least 200 examples per label")
        pool.sort(key=lambda row: hashlib.sha256(f"42:{row[0]}".encode()).digest())
        selected.extend(pool[:200])
    return sorted(selected, key=lambda row: row[0])


def ingest(session: Session, request: DatasetCreate) -> Dataset:
    existing = session.scalar(
        select(Dataset).where(
            Dataset.hf_dataset == request.hf_dataset,
            Dataset.hf_config == request.hf_config,
            Dataset.split == request.split,
            Dataset.hf_revision == request.hf_revision,
        )
    )
    if existing:
        return existing
    from datasets import load_dataset

    source = load_dataset(
        request.hf_dataset, request.hf_config, split=request.split, revision=request.hf_revision
    )
    provenance = {}
    if request.hf_dataset == "stanfordnlp/imdb":
        rows = imdb_sample(
            [(idx, str(row["text"]), int(row["label"])) for idx, row in enumerate(source)]
        )
        provenance = {
            "selection": "stratified-sha256-v1",
            "seed": 42,
            "per_label": 200,
            "source_n": len(source),
            "source_indices": [row[0] for row in rows],
        }
    else:
        rows = [(int(row["idx"]), str(row["sentence"]), int(row["label"])) for row in source]
    rows.sort(key=lambda row: row[0])
    if not rows or any(label not in (0, 1) for _, _, label in rows):
        raise ValueError("Dataset must have binary labels and at least one example")
    dataset = Dataset(
        **request.model_dump(),
        content_hash=content_hash(rows),
        n_examples=len(rows),
        provenance=provenance,
    )
    session.add(dataset)
    session.flush()
    session.add_all(
        [
            Example(dataset_id=dataset.id, idx=idx, text=text, label=label, slices=slice_tags(text))
            for idx, text, label in rows
        ]
    )
    session.commit()
    return dataset
