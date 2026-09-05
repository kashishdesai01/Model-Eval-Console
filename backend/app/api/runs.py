from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.errors import Problem
from app.api.registry import Database
from app.api.schemas import PredictionOut, RunCreate, RunOut
from app.db.models import Candidate, Dataset, EvalRun, Example, Prediction

router = APIRouter(prefix="/v1/runs", tags=["runs"])


def run_response(session: Database, run: EvalRun) -> RunOut:
    candidate, dataset = (
        session.get(Candidate, run.candidate_id),
        session.get(Dataset, run.dataset_id),
    )
    result = RunOut.model_validate(run)
    result.progress = (
        session.scalar(
            select(func.count())
            .select_from(Prediction)
            .where(
                Prediction.run_id == run.id,
            )
        )
        or 0
    )
    result.total = dataset.n_examples if dataset else 0
    result.candidate_name = candidate.name if candidate else "Unknown"
    result.dataset_name = dataset.name if dataset else "Unknown"
    return result


@router.post("", response_model=RunOut, status_code=202)
def enqueue_run(request: RunCreate, session: Database) -> RunOut:
    if (
        session.get(Candidate, request.candidate_id) is None
        or session.get(Dataset, request.dataset_id) is None
    ):
        raise Problem(404, "resource-not-found", "Candidate or dataset does not exist.")
    run = EvalRun(**request.model_dump())
    session.add(run)
    session.commit()
    return run_response(session, run)


@router.get("", response_model=list[RunOut])
def list_runs(session: Database, limit: int = Query(50, ge=1, le=100)) -> list[RunOut]:
    runs = session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit))
    return [run_response(session, run) for run in runs]


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: UUID, session: Database) -> RunOut:
    run = session.get(EvalRun, run_id)
    if run is None:
        raise Problem(404, "run-not-found", "Run does not exist.")
    return run_response(session, run)


@router.get("/{run_id}/predictions", response_model=list[PredictionOut])
def predictions(
    run_id: UUID,
    session: Database,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    correct: bool | None = None,
    slice: str | None = None,
) -> list[PredictionOut]:
    run = session.get(EvalRun, run_id)
    if run is None:
        raise Problem(404, "run-not-found", "Run does not exist.")
    query = (
        select(Prediction, Example)
        .join(
            Example,
            (Example.dataset_id == run.dataset_id) & (Example.idx == Prediction.example_idx),
        )
        .where(Prediction.run_id == run_id)
    )
    if correct is not None:
        query = query.where(Prediction.correct == correct)
    if slice:
        query = query.where(Example.slices.contains([slice]))
    rows = session.execute(query.order_by(Prediction.example_idx).offset(offset).limit(limit))
    return [
        PredictionOut(
            idx=e.idx,
            text=e.text,
            label=e.label,
            pred=p.pred,
            prob_pos=p.prob_pos,
            raw_output=p.raw_output,
            correct=p.correct,
            slices=e.slices,
        )
        for p, e in rows
    ]
