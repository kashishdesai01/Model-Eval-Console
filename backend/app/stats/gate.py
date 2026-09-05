from dataclasses import asdict, dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binomtest

from app.stats.bootstrap import paired_interval

Verdict = Literal["pass", "fail", "inconclusive"]


def verdict(low: float, high: float, margin: float) -> Verdict:
    if not 0 < margin <= 1 or low > high:
        raise ValueError("Invalid interval or non-inferiority margin")
    if low > -margin:
        return "pass"
    if high < -margin:
        return "fail"
    return "inconclusive"


@dataclass(frozen=True)
class GateResult:
    delta: float
    ci95: tuple[float, float]
    verdict: Verdict
    mcnemar_p: float
    regressions: int
    fixes: int
    standard_error: float
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    margin: float = 0.01,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> GateResult:
    interval = paired_interval(baseline, candidate, n_resamples=n_resamples, seed=seed)
    b, c = np.asarray(baseline), np.asarray(candidate)
    regressions = int(((b == 1) & (c == 0)).sum())
    fixes = int(((b == 0) & (c == 1)).sum())
    discordant = regressions + fixes
    p = float(binomtest(regressions, discordant, p=0.5).pvalue) if discordant else 1.0
    result = verdict(interval.low, interval.high, margin)
    warnings: list[str] = []
    if discordant < 20:
        warnings.append(
            "Sparse disagreements: percentile-bootstrap coverage can be poor. "
            "This gate does not establish equivalence outside this benchmark.",
        )
    if result == "inconclusive":
        warnings.append(
            "Interval crosses the margin; collect more data or explicitly accept the risk."
        )
    if result == "pass" and interval.high < 0:
        warnings.append("Statistically worse on this benchmark, but within the selected margin.")
    excludes_zero = interval.low > 0 or interval.high < 0
    if excludes_zero != (p < 0.05):
        warnings.append(
            "Bootstrap and McNemar disagree about zero difference; inspect sparse pairs."
        )
    variance = max(0.0, discordant / b.size - interval.delta**2)
    return GateResult(
        interval.delta,
        (interval.low, interval.high),
        result,
        p,
        regressions,
        fixes,
        sqrt(variance / b.size),
        warnings,
    )
