import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binomtest

from app.stats.bootstrap import paired_interval
from app.stats.gate import compare, verdict


def test_hand_computed_and_reproducible():
    b = [1, 1, 0, 0] * 100
    assert paired_interval(b, b).delta == 0
    assert paired_interval(b, b).low == paired_interval(b, b).high == 0
    result = paired_interval([1] * 100, [0] * 100)
    assert (result.delta, result.low, result.high) == (-1, -1, -1)
    assert paired_interval(b, [0, 1, 1, 0] * 100, seed=4) == paired_interval(
        b, [0, 1, 1, 0] * 100, seed=4
    )


@pytest.mark.parametrize(
    "low,high,margin,expected",
    [
        (-0.009, 0.001, 0.01, "pass"),
        (-0.02, -0.011, 0.01, "fail"),
        (-0.02, 0.0, 0.01, "inconclusive"),
        (-0.01, 0, 0.01, "inconclusive"),
        (-0.02, -0.01, 0.01, "inconclusive"),
    ],
)
def test_verdict_boundaries(low, high, margin, expected):
    assert verdict(low, high, margin) == expected


@given(st.lists(st.tuples(st.integers(0, 1), st.integers(0, 1)), min_size=1, max_size=100))
def test_paired_properties(pairs):
    b, c = zip(*pairs, strict=True)
    forward = paired_interval(b, c, n_resamples=1000)
    reverse = paired_interval(c, b, n_resamples=1000)
    assert -1 <= forward.low <= forward.high <= 1
    assert forward.delta == -reverse.delta
    # Finite Monte Carlo quantiles are not required to mirror exactly.
    assert abs(forward.low + reverse.high) <= 0.2


@pytest.mark.parametrize("b,c", [([], []), ([1], [1, 0]), ([2], [1]), ([[1]], [[0]])])
def test_invalid_inputs(b, c):
    with pytest.raises(ValueError):
        paired_interval(b, c)


def test_mcnemar_and_sparse_warning():
    b, c = [1] * 15 + [0] * 5, [0] * 15 + [1] * 5
    result = compare(b, c)
    assert result.mcnemar_p == binomtest(15, 20, p=0.5).pvalue
    assert compare([1] * 100, [1] * 100).warnings


@pytest.mark.slow
@pytest.mark.parametrize("probabilities", [(0.04, 0.92, 0.04), (0.03, 0.95, 0.02)])
def test_coverage_in_validated_regimes(probabilities):
    rng = np.random.default_rng(71)
    covered = 0
    delta = probabilities[2] - probabilities[0]
    for _ in range(500):
        d = rng.choice([-1, 0, 1], size=872, p=probabilities)
        interval = paired_interval(d <= 0, d >= 0, n_resamples=10_000, seed=42)
        covered += interval.low <= delta <= interval.high
    assert 0.91 <= covered / 500 <= 0.99


@pytest.mark.slow
def test_sparse_coverage_limitation_is_real():
    rng = np.random.default_rng(19)
    covered = 0
    for _ in range(500):
        c = rng.binomial(1, 0.001, size=872)
        interval = paired_interval(np.zeros(872, dtype=int), c, n_resamples=1000)
        covered += interval.low <= 0.001 <= interval.high
    assert covered / 500 < 0.8  # Demonstrates a limitation, not a universal coverage promise.
