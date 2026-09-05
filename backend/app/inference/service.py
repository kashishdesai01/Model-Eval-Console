import io
import platform
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from app.api.schemas import ServingConfig
from app.core.config import get_settings
from app.core.metrics import INFERENCE_LATENCY
from app.inference.generative import generate_sentiment, prompt_hash


@dataclass(frozen=True)
class BatchResult:
    labels: list[int]
    probabilities: list[float | None]
    latency_ms: float
    cold_start: bool
    raw_outputs: Sequence[str | None] | None = None


class InferenceService:
    """CPU-only, bounded model cache. Lock also serializes cache eviction and inference."""

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str, str], tuple[Any, Any, int]] = OrderedDict()
        self._lock = threading.RLock()
        torch.set_num_threads(get_settings().torch_threads)
        engines = torch.backends.quantized.supported_engines
        engine = next((name for name in ("x86", "fbgemm", "qnnpack") if name in engines), None)
        if engine is not None:
            torch.backends.quantized.engine = engine

    def _load(self, model_id: str, revision: str, config: ServingConfig) -> tuple[Any, Any, int]:
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model_id, revision=revision, trust_remote_code=False
        )
        loader = (
            AutoModelForCausalLM
            if config.inference_kind == "generative"
            else AutoModelForSequenceClassification
        )
        model = (
            loader.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
            )
            .cpu()
            .eval()
        )
        if config.inference_kind == "classifier" and model.config.num_labels != 2:
            raise ValueError("Only binary sequence-classification checkpoints are supported")
        if config.quantization == "int8":
            if torch.backends.quantized.engine == "none":
                raise ValueError("This PyTorch build has no supported CPU int8 engine")
            model = torch.ao.quantization.quantize_dynamic(  # type: ignore[no-untyped-call]
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
        if config.inference_kind == "generative":
            # Avoid allocating a second multi-GB copy just to measure weight storage.
            return (
                tokenizer,
                model,
                sum(t.numel() * t.element_size() for t in model.parameters()),
            )
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return tokenizer, model, buffer.tell()

    def predict(
        self,
        model_id: str,
        revision: str,
        config: ServingConfig,
        texts: list[str],
    ) -> BatchResult:
        if not texts:
            raise ValueError("At least one text is required")
        key = (model_id, revision, f"{config.inference_kind}:{config.quantization}:{config.dtype}")
        with self._lock:
            cold = key not in self._cache
            if cold:
                if config.inference_kind == "generative":
                    self._cache.clear()
                if len(self._cache) >= 2:
                    self._cache.popitem(last=False)
                self._cache[key] = self._load(model_id, revision, config)
            self._cache.move_to_end(key)
            tokenizer, model, _ = self._cache[key]
            started = time.perf_counter()
            if config.inference_kind == "generative":
                labels, outputs = generate_sentiment(tokenizer, model, config, texts)
                elapsed = (time.perf_counter() - started) * 1000
                INFERENCE_LATENCY.labels(config.quantization).observe(elapsed / 1000)
                return BatchResult(labels, [None] * len(texts), elapsed, cold, outputs)
            labels, probabilities = [], []
            for offset in range(0, len(texts), config.batch_size):
                inputs = tokenizer(
                    texts[offset : offset + config.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                )
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    positive = logits.softmax(dim=-1)[:, config.positive_label_id]
                    predicted = (logits.argmax(dim=-1) == config.positive_label_id).int()
                labels.extend(predicted.tolist())
                probabilities.extend(positive.tolist())
            elapsed = (time.perf_counter() - started) * 1000
            INFERENCE_LATENCY.labels(config.quantization).observe(elapsed / 1000)
            return BatchResult(labels, probabilities, elapsed, cold)

    def environment(self) -> dict[str, Any]:
        return {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "num_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "quantization_engine": torch.backends.quantized.engine,
            "evidence": "measured",
        }

    def benchmark(
        self,
        model_id: str,
        revision: str,
        config: ServingConfig,
        texts: list[str],
    ) -> dict[str, Any]:
        # Fixed, evenly spaced subset; same indices and batch size across candidates.
        indices = np.linspace(0, len(texts) - 1, min(16, len(texts)), dtype=int)
        sample = [texts[int(i)] for i in indices]
        batches = [
            sample[i : i + config.batch_size] for i in range(0, len(sample), config.batch_size)
        ]
        for i in range(5):
            self.predict(model_id, revision, config, batches[i % len(batches)])
        p95s, p50s = [], []
        for _ in range(3):
            times = [
                self.predict(model_id, revision, config, batch).latency_ms for batch in batches
            ]
            p95s.append(float(np.quantile(times, 0.95)))
            p50s.append(float(np.median(times)))
        key = (model_id, revision, f"{config.inference_kind}:{config.quantization}:{config.dtype}")
        return {
            "latency_p95_ms": float(np.median(p95s)),
            "latency_p50_ms": float(np.median(p50s)),
            "weights_bytes": self._cache[key][2],
            "weights_size_method": "tensor-storage"
            if config.inference_kind == "generative"
            else "serialized-state-dict",
            "prompt_sha256": prompt_hash(config.prompt_version)
            if config.inference_kind == "generative"
            else None,
            "latency_batch_size": config.batch_size,
            "latency_sample_indices": indices.tolist(),
            "latency_repeats": 3,
            "latency_warmup_batches": 5,
        }
