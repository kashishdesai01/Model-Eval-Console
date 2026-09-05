from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Interval:
    delta: float
    low: float
    high: float


def paired_interval(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Interval:
    b, c = np.asarray(baseline), np.asarray(candidate)
    if b.ndim != 1 or c.shape != b.shape or not b.size:
        raise ValueError("Expected nonempty paired vectors of equal length")
    if not np.isin(b, [0, 1]).all() or not np.isin(c, [0, 1]).all():
        raise ValueError("Correctness vectors must contain only 0 or 1")
    if not 100 <= n_resamples <= 50_000:
        raise ValueError("Resample count must be between 100 and 50000")
    d = c.astype(np.int8) - b.astype(np.int8)
    # A multinomial draw of the {-1,0,+1} counts is exactly an empirical
    # index bootstrap for their mean, with O(B) rather than O(B*n) memory.
    counts = np.array([(d == -1).sum(), (d == 0).sum(), (d == 1).sum()])
    draws = np.random.default_rng(seed).multinomial(d.size, counts / d.size, size=n_resamples)
    means = (draws[:, 2] - draws[:, 0]) / d.size
    low, high = np.quantile(means, [0.025, 0.975])
    return Interval(float(d.mean()), float(low), float(high))
