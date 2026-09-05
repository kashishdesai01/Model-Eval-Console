from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.errors import Problem
from app.api.schemas import ComparisonCreate, ComparisonOut
from app.db.models import Candidate, Comparison, Dataset, EvalRun, Example, Prediction
from app.stats.gate import compare


def create_comparison(session: Session, request: ComparisonCreate) -> Comparison:
    baseline = session.get(EvalRun, request.baseline_run_id)
    candidate = session.get(EvalRun, request.candidate_run_id)
    if baseline is None or candidate is None:
        raise Problem(404, "run-not-found", "A selected run does not exist.")
    if baseline.status != "succeeded" or candidate.status != "succeeded":
        raise Problem(422, "run-incomplete", "Both selected runs must have succeeded.")
    if baseline.dataset_id != candidate.dataset_id:
        raise Problem(
            422, "dataset-mismatch", "Runs must use the same immutable, content-hashed dataset."
        )
    dataset = session.get(Dataset, baseline.dataset_id)
    assert dataset is not None
    examples = list(
        session.scalars(
            select(Example)
            .where(
                Example.dataset_id == dataset.id,
            )
            .order_by(Example.idx)
        )
    )
    bp = list(
        session.scalars(
            select(Prediction)
            .where(
                Prediction.run_id == baseline.id,
            )
            .order_by(Prediction.example_idx)
        )
    )
    cp = list(
        session.scalars(
            select(Prediction)
            .where(
                Prediction.run_id == candidate.id,
            )
            .order_by(Prediction.example_idx)
        )
    )
    expected = [e.idx for e in examples]
    if (
        len(expected) != dataset.n_examples
        or [p.example_idx for p in bp] != expected
        or [p.example_idx for p in cp] != expected
    ):
        raise Problem(
            422, "prediction-mismatch", "Prediction IDs do not exactly match the dataset."
        )
    b, c = np.array([p.correct for p in bp]), np.array([p.correct for p in cp])
    gate = compare(b, c, margin=request.margin, n_resamples=request.n_resamples, seed=request.seed)
    slices = []
    for tag in sorted({tag for example in examples for tag in example.slices}):
        mask = np.array([tag in example.slices for example in examples])
        if mask.sum() < 50:
            continue
        result = compare(
            b[mask],
            c[mask],
            margin=request.margin,
            n_resamples=request.n_resamples,
            seed=request.seed,
        )
        slices.append(
            {"name": tag, "n": int(mask.sum()), "delta": result.delta, "ci95": result.ci95}
        )
    flips = []
    for example, before, after in zip(examples, bp, cp, strict=True):
        if before.correct == after.correct:
            continue
        flips.append(
            {
                "idx": example.idx,
                "text": example.text,
                "label": example.label,
                "baseline_prob": before.prob_pos,
                "candidate_prob": after.prob_pos,
                "baseline_output": before.raw_output,
                "candidate_output": after.raw_output,
                "baseline_pred": before.pred,
                "candidate_pred": after.pred,
                "slices": example.slices,
                "near_boundary": any(
                    abs(p - 0.5) <= 0.01 for p in (before.prob_pos, after.prob_pos) if p is not None
                ),
                "kind": "regression" if before.correct else "fix",
            }
        )
    regressions = [flip for flip in flips if flip["kind"] == "regression"]
    near_share = sum(bool(f["near_boundary"]) for f in regressions) / max(1, len(regressions))
    baseline_model = session.get(Candidate, baseline.candidate_id)
    candidate_model = session.get(Candidate, candidate.candidate_id)
    assert baseline_model is not None and candidate_model is not None
    ratio = None
    bm, cm = baseline.metrics or {}, candidate.metrics or {}
    if (
        baseline.env == candidate.env
        and bm.get("latency_batch_size") == cm.get("latency_batch_size")
        and bm.get("latency_sample_indices") == cm.get("latency_sample_indices")
        and bm.get("latency_p95_ms", 0) > 0
        and cm.get("latency_p95_ms", 0) > 0
    ):
        ratio = cm["latency_p95_ms"] / bm["latency_p95_ms"]
    warnings = list(gate.warnings)
    for role, predictions in (("Baseline", bp), ("Candidate", cp)):
        invalid = sum(p.pred == -1 for p in predictions)
        if invalid:
            warnings.append(
                f"{role}: {invalid} invalid outputs count as incorrect; none were dropped."
            )
    if baseline.code_version != candidate.code_version or baseline.env != candidate.env:
        warnings.append(
            "Evaluator code or environment differs; quality comparison has a confounder."
        )
    evidence = (
        "fixture"
        if "fixture" in (baseline.env.get("evidence"), candidate.env.get("evidence"))
        else "measured"
    )
    detail: dict[str, Any] = {
        "baseline": {
            "run_id": str(baseline.id),
            "name": baseline_model.name,
            "value": float(b.mean()),
        },
        "candidate": {
            "run_id": str(candidate.id),
            "name": candidate_model.name,
            "value": float(c.mean()),
        },
        "regressions": gate.regressions,
        "fixes": gate.fixes,
        "standard_error": gate.standard_error,
        "warnings": warnings,
        "slices": slices,
        "flips": flips,
        "near_boundary_share": near_share,
        "n_examples": len(examples),
        "latency_ratio": ratio,
        "latency_budget_pass": ratio <= 1.10 if ratio is not None else None,
        "evidence": evidence,
        "dataset_content_hash": dataset.content_hash,
    }
    comparison = Comparison(
        **request.model_dump(),
        delta=gate.delta,
        ci_low=gate.ci95[0],
        ci_high=gate.ci95[1],
        verdict=gate.verdict,
        mcnemar_p=gate.mcnemar_p,
        detail=detail,
    )
    session.add(comparison)
    session.commit()
    return comparison


def comparison_response(comparison: Comparison) -> ComparisonOut:
    return ComparisonOut(
        id=comparison.id,
        metric=comparison.metric,
        margin=comparison.margin,
        delta=comparison.delta,
        ci95=(comparison.ci_low, comparison.ci_high),
        verdict=comparison.verdict,
        mcnemar_p=comparison.mcnemar_p,
        seed=comparison.seed,
        n_resamples=comparison.n_resamples,
        **{key: value for key, value in comparison.detail.items() if key != "dataset_content_hash"},
    )


def get_comparison(session: Session, comparison_id: UUID) -> Comparison:
    comparison = session.get(Comparison, comparison_id)
    if comparison is None:
        raise Problem(404, "comparison-not-found", "Comparison does not exist.")
    return comparison
