from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
HubId = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[\w.-]+(/[\w.-]+)?$")]


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_length: int = Field(default=128, ge=4, le=512)
    quantization: Literal["none", "int8"] = "none"
    batch_size: int = Field(default=8, ge=1, le=64)
    positive_label_id: Literal[0, 1] = 1
    inference_kind: Literal["classifier", "generative"] = "classifier"
    dtype: Literal["float32", "bfloat16"] = "float32"
    prompt_version: Literal["sentiment-zero-v1", "sentiment-few-v1"] = "sentiment-zero-v1"
    max_new_tokens: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def supported_generation(self) -> "ServingConfig":
        if self.inference_kind == "classifier" and self.dtype != "float32":
            raise ValueError("Classifier inference requires float32 weights")
        if self.inference_kind == "generative" and (
            self.quantization != "none" or self.batch_size != 1 or self.positive_label_id != 1
        ):
            raise ValueError(
                "Generative inference requires no quantization, batch size 1, mapping 1"
            )
        return self


class CandidateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[\w .-]+$")
    hf_model_id: HubId
    hf_revision: Revision
    serving_config: ServingConfig = Field(default_factory=ServingConfig)


class CandidateOut(CandidateCreate):
    model_config = ConfigDict(from_attributes=True)
    serving_config: ServingConfig
    id: UUID
    created_at: datetime


class DatasetCreate(BaseModel):
    name: str = Field(default="SST-2 validation", min_length=1, max_length=100)
    hf_dataset: Literal["stanfordnlp/sst2", "stanfordnlp/imdb"] = "stanfordnlp/sst2"
    hf_config: Literal["default", "plain_text"] = "default"
    split: Literal["validation", "test"] = "validation"
    hf_revision: Revision

    @model_validator(mode="after")
    def supported_split(self) -> "DatasetCreate":
        expected = (
            ("default", "validation")
            if self.hf_dataset.endswith("sst2")
            else ("plain_text", "test")
        )
        if (self.hf_config, self.split) != expected:
            raise ValueError("Use SST-2 default/validation or IMDb plain_text/test")
        return self


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    hf_dataset: str
    hf_config: str
    split: str
    hf_revision: str
    content_hash: str
    n_examples: int
    provenance: dict[str, Any] = Field(default_factory=dict)


class RunCreate(BaseModel):
    candidate_id: UUID
    dataset_id: UUID


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    candidate_id: UUID
    dataset_id: UUID
    status: Literal["queued", "running", "succeeded", "failed"]
    attempts: int
    max_attempts: int
    code_version: str
    env: dict[str, Any]
    metrics: dict[str, Any] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: int = 0
    total: int = 0
    candidate_name: str = ""
    dataset_name: str = ""


class ComparisonCreate(BaseModel):
    baseline_run_id: UUID
    candidate_run_id: UUID
    metric: Literal["accuracy"] = "accuracy"
    margin: float = Field(default=0.01, gt=0, le=1)
    n_resamples: int = Field(default=10_000, ge=100, le=50_000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class ScoreRef(BaseModel):
    run_id: UUID
    name: str
    value: float


class SliceResult(BaseModel):
    name: str
    n: int
    delta: float
    ci95: tuple[float, float]


class Flip(BaseModel):
    idx: int
    text: str
    label: int
    baseline_prob: float | None
    candidate_prob: float | None
    baseline_output: str | None = None
    candidate_output: str | None = None
    baseline_pred: int | None = None
    candidate_pred: int | None = None
    slices: list[str]
    near_boundary: bool
    kind: Literal["regression", "fix"]


class ComparisonOut(BaseModel):
    id: UUID
    metric: str
    baseline: ScoreRef
    candidate: ScoreRef
    margin: float
    delta: float
    ci95: tuple[float, float]
    verdict: Literal["pass", "fail", "inconclusive"]
    mcnemar_p: float
    regressions: int
    fixes: int
    standard_error: float
    warnings: list[str]
    slices: list[SliceResult]
    flips: list[Flip]
    near_boundary_share: float
    n_examples: int
    seed: int
    n_resamples: int
    latency_budget_pass: bool | None
    latency_ratio: float | None
    evidence: Literal["measured", "fixture"]


class PredictionOut(BaseModel):
    idx: int
    text: str
    label: int
    pred: int
    prob_pos: float | None
    raw_output: str | None = None
    correct: bool
    slices: list[str]


class PredictRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=20_000)]] = Field(
        min_length=1,
        max_length=64,
    )


class OnlinePrediction(BaseModel):
    label: int
    prob_pos: float | None
    raw_output: str | None = None


class PredictResponse(BaseModel):
    predictions: list[OnlinePrediction]
    latency_ms: float
    cold_start: bool
