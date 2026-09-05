from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.errors import Problem
from app.api.schemas import CandidateCreate, CandidateOut, DatasetCreate, DatasetOut
from app.db.models import Candidate, Dataset
from app.db.session import get_session
from app.services.datasets import ingest

router = APIRouter(prefix="/v1", tags=["registry"])
Database = Annotated[Session, Depends(get_session)]


@router.post("/candidates", response_model=CandidateOut, status_code=201)
def register_candidate(request: CandidateCreate, session: Database) -> Candidate:
    candidate = Candidate(**request.model_dump())
    session.add(candidate)
    session.commit()
    return candidate


@router.get("/candidates", response_model=list[CandidateOut])
def list_candidates(
    session: Database, limit: int = Query(100, ge=1, le=100), after: UUID | None = None
) -> list[Candidate]:
    query = select(Candidate).order_by(Candidate.id).limit(limit)
    if after:
        query = query.where(Candidate.id > after)
    return list(session.scalars(query))


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(session: Database) -> list[Dataset]:
    return list(session.scalars(select(Dataset).order_by(Dataset.created_at).limit(100)))


@router.post("/datasets", response_model=DatasetOut, status_code=201)
def ingest_dataset(request: DatasetCreate, session: Database) -> Dataset:
    try:
        return ingest(session, request)
    except (ValueError, OSError) as exc:
        raise Problem(422, "dataset-ingestion-failed", str(exc)) from exc
