from uuid import UUID

from fastapi import APIRouter, Request

from app.api.errors import Problem
from app.api.registry import Database
from app.api.schemas import OnlinePrediction, PredictRequest, PredictResponse, ServingConfig
from app.core.metrics import PREDICT_REQUESTS
from app.db.models import Candidate

router = APIRouter(prefix="/v1/models", tags=["inference"])


@router.post("/{candidate_id}/predict", response_model=PredictResponse)
def predict(
    candidate_id: UUID,
    body: PredictRequest,
    request: Request,
    session: Database,
) -> PredictResponse:
    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        raise Problem(404, "candidate-not-found", "Candidate does not exist.")
    try:
        result = request.app.state.inference.predict(
            candidate.hf_model_id,
            candidate.hf_revision,
            ServingConfig.model_validate(candidate.serving_config),
            body.texts,
        )
    except (ValueError, OSError) as exc:
        PREDICT_REQUESTS.labels("error").inc()
        raise Problem(422, "inference-failed", str(exc)) from exc
    PREDICT_REQUESTS.labels("success").inc()
    return PredictResponse(
        predictions=[
            OnlinePrediction(label=label, prob_pos=prob, raw_output=output)
            for label, prob, output in zip(
                result.labels,
                result.probabilities,
                result.raw_outputs or [None] * len(result.labels),
                strict=True,
            )
        ],
        latency_ms=result.latency_ms,
        cold_start=result.cold_start,
    )
