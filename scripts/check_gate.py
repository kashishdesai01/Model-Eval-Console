"""Assert the CLI's real process exit codes against a running fixture API."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fixtures", default="artifacts/fixture-results.json")
parser.add_argument("--api-url", default="http://localhost:8000")
args = parser.parse_args()
runs = json.loads(Path(args.fixtures).read_text())["runs"]
env = {**os.environ, "MEC_API_URL": args.api_url}
for name, expected in [
    ("fixture-noop", 0),
    ("fixture-borderline", 2),
    ("fixture-truncation", 1),
    ("missing-run", 3),
]:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli.main",
            "gate",
            "--baseline",
            runs["fixture-baseline"],
            "--candidate",
            runs.get(name, "00000000-0000-0000-0000-000000000000"),
            "--api-url",
            args.api_url,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert process.returncode == expected, (name, process.returncode, process.stderr)
    print(f"{name}: exit {expected}")
