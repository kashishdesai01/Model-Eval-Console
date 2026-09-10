"""Record empirical coverage in explicitly scoped synthetic regimes."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.stats.bootstrap import paired_interval  # noqa: E402

regimes = {
    "balanced": (0.04, 0.92, 0.04),
    "asymmetric": (0.03, 0.95, 0.02),
    "sparse_improvement": (0.0, 0.999, 0.001),
}
results = {}
for name, probabilities in regimes.items():
    rng = np.random.default_rng(71)
    delta = probabilities[2] - probabilities[0]
    covered = 0
    for _ in range(500):
        d = rng.choice([-1, 0, 1], size=872, p=probabilities)
        interval = paired_interval(d <= 0, d >= 0, n_resamples=10_000, seed=42)
        covered += interval.low <= delta <= interval.high
    results[name] = {
        "probabilities_minus_zero_plus": probabilities,
        "true_delta": delta,
        "trials": 500,
        "n_examples": 872,
        "covered": covered,
        "coverage": covered / 500,
        "data_seed": 71,
        "bootstrap_seed": 42,
        "n_resamples": 10_000,
    }
output = Path(__file__).resolve().parents[1] / "docs/results/coverage.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
