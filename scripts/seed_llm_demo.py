"""Frozen classifier/LLM comparison on the fixed IMDb test sample; public API only."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

from app.api.schemas import ServingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", default="artifacts/llm-results.json")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    precision = "fp32" if args.dtype == "float32" else "bf16"
    zero_name, few_name = f"qwen-zero-{precision}-v1", f"qwen-few-{precision}-v1"
    manifest = json.loads(Path(__file__).with_name("llm_manifest.json").read_text())
    baseline = json.loads(Path(__file__).with_name("demo_manifest.json").read_text())
    with httpx.Client(base_url=args.api_url, timeout=600) as client:

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()

        dataset = post(
            "/v1/datasets",
            {
                "name": "IMDb test · balanced 400 · seed 42",
                "hf_dataset": manifest["dataset_id"],
                "hf_config": manifest["dataset_config"],
                "split": "test",
                "hf_revision": manifest["dataset_revision"],
            },
        )
        response = client.get("/v1/candidates")
        response.raise_for_status()
        existing = {c["name"]: c for c in response.json()}
        configs = {
            "distilbert-review128": (baseline, ServingConfig(batch_size=1, max_length=128)),
            zero_name: (
                manifest,
                ServingConfig(
                    inference_kind="generative", dtype=args.dtype, batch_size=1, max_length=128
                ),
            ),
            few_name: (
                manifest,
                ServingConfig(
                    inference_kind="generative",
                    dtype=args.dtype,
                    batch_size=1,
                    max_length=128,
                    prompt_version="sentiment-few-v1",
                ),
            ),
        }
        runs = {}
        candidates = {}
        for name, (model, config) in configs.items():
            body = {
                "name": name,
                "hf_model_id": model["model_id"],
                "hf_revision": model["model_revision"],
                "serving_config": config.model_dump(),
            }
            candidate = existing.get(name) or post("/v1/candidates", body)
            if any(candidate[key] != value for key, value in body.items()):
                raise ValueError(f"Existing candidate {name} differs from the frozen experiment")
            candidates[name] = candidate
            runs[name] = post(
                "/v1/runs", {"candidate_id": candidate["id"], "dataset_id": dataset["id"]}
            )
            print(f"Enqueued {name}: {runs[name]['id']}", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            for name, run in runs.items():
                response = client.get(f"/v1/runs/{run['id']}")
                response.raise_for_status()
                runs[name] = response.json()
                if runs[name]["status"] == "failed":
                    raise RuntimeError(f"{name}: {runs[name]['error']}")
            print(
                " | ".join(
                    f"{name}: {run['status']} {run['progress']}/{run['total']}"
                    for name, run in runs.items()
                ),
                flush=True,
            )
            if all(run["status"] == "succeeded" for run in runs.values()):
                break
            time.sleep(10)
        else:
            raise TimeoutError("LLM demo did not complete")
        comparisons = {}
        for name in (zero_name, few_name):
            comparisons[name] = post(
                "/v1/comparisons",
                {
                    "baseline_run_id": runs["distilbert-review128"]["id"],
                    "candidate_run_id": runs[name]["id"],
                    "margin": 0.01,
                    "n_resamples": 10_000,
                    "seed": 42,
                },
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions = {}
        for name, run in runs.items():
            records = []
            for offset in range(0, run["total"], 100):
                response = client.get(
                    f"/v1/runs/{run['id']}/predictions", params={"offset": offset, "limit": 100}
                )
                response.raise_for_status()
                records.extend(
                    {
                        key: row[key]
                        for key in ("idx", "label", "pred", "correct", "prob_pos", "raw_output")
                    }
                    for row in response.json()
                )
            if len(records) != dataset["n_examples"]:
                raise ValueError("Incomplete prediction export")
            predictions[name] = records
        output.write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "candidates": candidates,
                    "dataset": dataset,
                    "runs": runs,
                    "comparisons": comparisons,
                    "predictions": predictions,
                },
                indent=2,
            )
            + "\n"
        )
        for name, result in comparisons.items():
            print(
                f"{name}: {result['verdict'].upper()}, delta={result['delta']:.5f}, "
                f"CI={result['ci95']}"
            )
        print(f"Results: {output}")
        print(f"Console: http://localhost:5173/compare?comparison={comparisons[few_name]['id']}")


if __name__ == "__main__":
    main()
