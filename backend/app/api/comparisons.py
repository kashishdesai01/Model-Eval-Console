from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.registry import Database
from app.api.schemas import ComparisonCreate, ComparisonOut
from app.db.models import Comparison
from app.services.comparisons import comparison_response, create_comparison, get_comparison

router = APIRouter(prefix="/v1/comparisons", tags=["comparisons"])


@router.get("", response_model=list[ComparisonOut])
def list_comparisons(session: Database, limit: int = Query(5, ge=1, le=20)) -> list[ComparisonOut]:
    return [
        comparison_response(row)
        for row in session.scalars(
            select(Comparison).order_by(Comparison.created_at.desc()).limit(limit)
        )
    ]


@router.post("", response_model=ComparisonOut, status_code=201)
def compare_runs(request: ComparisonCreate, session: Database) -> ComparisonOut:
    return comparison_response(create_comparison(session, request))


@router.get("/{comparison_id}", response_model=ComparisonOut)
def read_comparison(comparison_id: UUID, session: Database) -> ComparisonOut:
    return comparison_response(get_comparison(session, comparison_id))
