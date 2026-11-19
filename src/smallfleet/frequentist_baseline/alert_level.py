"""The classical alert level of an airline reliability programme.

FAA AC 120-17B §4.1 asks for alert values "based on generally accepted statistical methods,
such as standard deviation or the Poisson distribution". Transport Canada AMA 571.101/1,
Appendix A, works three of them out: a multiple of the mean, the mean plus k standard
deviations of monthly (or 3-month running-average) rates over a 12-point baseline, with the
population divisor N, and the 95 % Poisson limit on the expected count of a period.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from smallfleet.data_generation import FloatArray


def mean_sd_alert_level(baseline_rates: FloatArray, k: float = 2.0, ddof: int = 0) -> float:
    """``mean + k·sd`` of the baseline rates (AMA 571.101/1 uses ``ddof = 0``)."""
    if baseline_rates.size < 2:
        raise ValueError("need at least two baseline periods")
    return float(np.mean(baseline_rates) + k * np.std(baseline_rates, ddof=ddof))


def mean_ratio_alert_level(baseline_rates: FloatArray, factor: float = 1.3) -> float:
    """``factor × mean`` of the baseline rates (AMA 571.101/1, example 1)."""
    return float(factor * np.mean(baseline_rates))


def poisson_alert_count(expected_count: float, probability: float = 0.95) -> int:
    """Smallest count ``A`` with ``P(Y ≤ A) ≥ probability`` for ``Y ~ Poisson(expected)``.

    An alert is raised when the observed count exceeds ``A``.
    """
    if expected_count < 0 or not 0 < probability < 1:
        raise ValueError("expected count must be non-negative and probability in (0, 1)")
    return int(stats.poisson.ppf(probability, expected_count))


def running_average(rates: FloatArray, window: int = 3) -> FloatArray:
    """Trailing running average; the first ``window − 1`` periods are dropped."""
    if window < 1 or rates.size < window:
        raise ValueError("window must be between 1 and the number of periods")
    kernel = np.ones(window) / window
    return np.convolve(rates, kernel, mode="valid")


def alert_flags(rates: FloatArray, level: float, consecutive: int = 1) -> np.ndarray:
    """Periods in which the last ``consecutive`` rates all exceed ``level``."""
    if consecutive < 1:
        raise ValueError("consecutive must be at least 1")
    above = rates > level
    if consecutive == 1:
        return above
    flags = np.zeros_like(above)
    for j in range(consecutive - 1, above.size):
        flags[j] = bool(np.all(above[j - consecutive + 1 : j + 1]))
    return flags
