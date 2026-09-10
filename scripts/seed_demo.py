"""Run the actual pinned model on the actual benchmark through the public API."""

import argparse
import json
import time
from pathlib import Path

import httpx

from app.api.schemas import ServingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", default="artifacts/demo-results.json")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("demo_manifest.json").read_text())
    with httpx.Client(base_url=args.api_url, timeout=600) as client:
        ready_deadline = time.monotonic() + 90
        while time.monotonic() < ready_deadline:
            try:
                if client.get("/readyz", timeout=3).is_success:
                    break
            except httpx.TransportError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("API did not become ready within 90 seconds")

        def post(path: str, body: dict):
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()

        dataset = post(
            "/v1/datasets",
            {
                "hf_revision": manifest["dataset_revision"],
                "hf_dataset": manifest["dataset_id"],
                "hf_config": manifest["dataset_config"],
                "split": "validation",
                "name": "SST-2 validation",
            },
        )
        registered_response = client.get("/v1/candidates")
        registered_response.raise_for_status()
        existing = {candidate["name"]: candidate for candidate in registered_response.json()}
        configs = {
            "baseline": (128, "none", 8),
            "batch32": (128, "none", 32),
            "int8": (128, "int8", 8),
            "trunc16": (16, "none", 8),
        }
        runs = {}
        for name, (max_length, quantization, batch_size) in configs.items():
            body = {
                "name": name,
                "hf_model_id": manifest["model_id"],
                "hf_revision": manifest["model_revision"],
                "serving_config": {
                    "max_length": max_length,
                    "quantization": quantization,
                    "batch_size": batch_size,
                    "positive_label_id": 1,
                },
            }
            body["serving_config"] = ServingConfig.model_validate(
                body["serving_config"]
            ).model_dump()
            candidate = existing.get(name) or post("/v1/candidates", body)
            if any(candidate[key] != body[key] for key in body):
                raise ValueError(f"Existing candidate {name} differs from the demo manifest")
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
                    raise RuntimeError(f"{name} failed: {runs[name]['error']}")
            print(
                " | ".join(
                    f"{name}: {run['status']} {run['progress']}/{run['total']}"
                    for name, run in runs.items()
                ),
                flush=True,
            )
            if all(run["status"] == "succeeded" for run in runs.values()):
                break
            time.sleep(5)
        else:
            raise TimeoutError("Demo runs did not finish before the timeout")
        comparisons = {}
        for name in ("batch32", "int8", "trunc16"):
            comparisons[name] = post(
                "/v1/comparisons",
                {
                    "baseline_run_id": runs["baseline"]["id"],
                    "candidate_run_id": runs[name]["id"],
                    "margin": 0.01,
                    "n_resamples": 10_000,
                    "seed": 42,
                },
            )
            result = comparisons[name]
            print(
                f"{name}: {result['verdict'].upper()}, delta={result['delta']:.5f}, "
                f"CI={result['ci95']}",
                flush=True,
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "dataset": dataset,
                    "runs": runs,
                    "comparisons": comparisons,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Measured results: {output.resolve()}")
        print(f"Console: http://localhost:5173/compare?comparison={comparisons['trunc16']['id']}")


if __name__ == "__main__":
    main()
