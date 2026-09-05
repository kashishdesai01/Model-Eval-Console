import pytest
import torch
from pydantic import ValidationError

from app.api.schemas import DatasetCreate, ServingConfig
from app.inference.generative import (
    SYSTEM,
    generate_sentiment,
    parse_label,
    prompt_hash,
    prompt_spec,
)
from app.services.datasets import imdb_sample
from app.worker.runner import quality_metrics


@pytest.mark.parametrize(
    "output,expected",
    [
        ("positive", 1),
        ("negative\n", 0),
        ("", -1),
        ("Positive", -1),
        ("positive.", -1),
        ("positive because it is good", -1),
        ("negative positive", -1),
    ],
)
def test_strict_parser(output, expected):
    assert parse_label(output) == expected


def test_prompt_identity_and_handwritten_examples():
    assert prompt_hash("sentiment-zero-v1") != prompt_hash("sentiment-few-v1")
    assert prompt_hash("sentiment-zero-v1") == prompt_hash("sentiment-zero-v1")
    assert len(prompt_spec("sentiment-few-v1")) == 5
    with pytest.raises(ValueError):
        prompt_spec("unknown")


def test_config_and_dataset_contracts():
    with pytest.raises(ValidationError):
        ServingConfig(dtype="bfloat16")
    assert (
        ServingConfig(inference_kind="generative", batch_size=1, dtype="bfloat16").dtype
        == "bfloat16"
    )
    with pytest.raises(ValidationError):
        ServingConfig(inference_kind="generative")  # batch 8 is unsupported
    with pytest.raises(ValidationError):
        ServingConfig(inference_kind="generative", batch_size=1, quantization="int8")
    with pytest.raises(ValidationError):
        DatasetCreate(hf_dataset="stanfordnlp/imdb", hf_revision="a" * 40)


def test_stratified_sample_is_reproducible_without_contiguous_ids():
    source = [(idx, f"Review {idx}", idx % 2) for idx in range(1000)]
    rows = imdb_sample(source)
    assert rows == imdb_sample(source[::-1])
    assert len(rows) == len({row[0] for row in rows}) == 400
    assert sum(row[2] for row in rows) == 200
    with pytest.raises(ValueError):
        imdb_sample(source[:100])


def test_invalid_outputs_count_as_errors_and_false_negatives():
    result = quality_metrics([0, 1, 0, 1], [-1, -1, 0, 1])
    assert result["accuracy"] == 0.5
    assert result["invalid_outputs"] == 2
    assert result["invalid_output_rate"] == 0.5
    assert result["macro_f1"] == pytest.approx(2 / 3)


def test_generation_preserves_instructions_and_caps_review_and_output():
    class Tokenizer:
        eos_token_id = 2
        messages = []

        def encode(self, text, **kwargs):
            return list(range(1, len(text.split()) + 1))

        def decode(self, ids, **kwargs):
            return (
                "positive" if len(ids) == 1 and int(ids[0]) == 0 else " ".join("word" for _ in ids)
            )

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            assert kwargs["enable_thinking"] is False
            return "chat"

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    class Generator:
        def __init__(self, token=0):
            self.token = token

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 8
            return torch.tensor([[1, 2, self.token]])

    tokenizer = Tokenizer()
    labels, outputs = generate_sentiment(
        tokenizer,
        Generator(),
        ServingConfig(inference_kind="generative", batch_size=1, max_length=4),
        ["word " * 100],
    )
    assert labels == [1] and outputs == ["positive"]
    assert tokenizer.messages[0]["content"] == SYSTEM
    assert tokenizer.messages[-1]["content"] == "Review: word word word word"
    labels, outputs = generate_sentiment(
        tokenizer,
        Generator(token=3),
        ServingConfig(inference_kind="generative", batch_size=1),
        ["review"],
    )
    assert labels == [-1] and outputs == ["word"]
