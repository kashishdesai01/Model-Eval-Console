from types import SimpleNamespace

import numpy as np
import torch

from app.api.schemas import ServingConfig
from app.inference.service import InferenceService


class TinyTokenizer:
    def __call__(self, texts, *, return_tensors, padding, truncation, max_length):
        lengths = [min(len(text.split()), max_length) for text in texts]
        return {"input_ids": torch.tensor(lengths).reshape(-1, 1).float()}


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[-1.0], [1.0]]))
            self.linear.bias.copy_(torch.tensor([2.5, -2.5]))

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.linear(input_ids))


def test_batch_single_truncation_and_actual_int8(monkeypatch):
    def load(_self, _id, _revision, config):
        model = TinyClassifier().eval()
        if config.quantization == "int8":
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        return TinyTokenizer(), model, 100

    monkeypatch.setattr(InferenceService, "_load", load)
    service = InferenceService()
    texts = ["one", "one two three four five"]
    fp32 = service.predict("test/model", "a" * 40, ServingConfig(), texts)
    singles = [service.predict("test/model", "a" * 40, ServingConfig(), [text]) for text in texts]
    assert fp32.labels == [result.labels[0] for result in singles] == [0, 1]
    assert np.allclose(fp32.probabilities, [result.probabilities[0] for result in singles])
    quantized = service.predict("test/model", "a" * 40, ServingConfig(quantization="int8"), texts)
    assert quantized.labels == fp32.labels
    assert service.predict(
        "test/model", "a" * 40, ServingConfig(positive_label_id=0), texts
    ).labels == [1, 0]
    assert len(service._cache) == 2
    truncated = service.predict("test/model", "a" * 40, ServingConfig(max_length=4), texts)
    assert truncated.probabilities[1] < fp32.probabilities[1]
