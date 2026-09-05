import argparse
import json
import sys

import httpx

from app.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark quality gate for model promotions")
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--metric", choices=["accuracy"], default="accuracy")
    gate.add_argument("--margin", type=float, default=0.01)
    gate.add_argument("--seed", type=int, default=42)
    gate.add_argument("--api-url", default=get_settings().api_url)
    args = parser.parse_args()
    try:
        response = httpx.post(
            f"{args.api_url}/v1/comparisons",
            timeout=60,
            json={
                "baseline_run_id": args.baseline,
                "candidate_run_id": args.candidate,
                "metric": args.metric,
                "margin": args.margin,
                "seed": args.seed,
            },
        )
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2))
        sys.exit({"pass": 0, "fail": 1, "inconclusive": 2}[result["verdict"]])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        print(f"Gate error: {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
