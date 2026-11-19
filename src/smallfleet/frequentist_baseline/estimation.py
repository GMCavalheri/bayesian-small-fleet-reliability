"""Classical point and interval estimates of a Poisson failure rate."""

from __future__ import annotations

import numpy as np
from scipy import stats

from smallfleet.data_generation import PER, FloatArray, IntArray


def mle_rate(count: int, exposure_fh: float) -> float:
    """Maximum-likelihood rate per 1000 FH: ``y / T``."""
    if exposure_fh <= 0 or count < 0:
        raise ValueError("exposure must be positive and the count non-negative")
    return PER * count / exposure_fh


def garwood_interval(
    count: int, exposure_fh: float, confidence: float = 0.90
) -> tuple[float, float]:
    """Exact (Garwood 1936) two-sided interval for a Poisson rate, per 1000 FH.

    ``L = χ²(α/2; 2y) / 2T`` and ``U = χ²(1 − α/2; 2y + 2) / 2T``, with ``L = 0`` when
    ``y = 0``. Its coverage is at least ``confidence`` for every true rate.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    mle_rate(count, exposure_fh)  # validates the inputs
    a = 1.0 - confidence
    t = exposure_fh / PER
    lower = 0.0 if count == 0 else float(stats.chi2.ppf(a / 2, 2 * count)) / (2 * t)
    upper = float(stats.chi2.ppf(1 - a / 2, 2 * count + 2)) / (2 * t)
    return lower, upper


def garwood_coverage(rate: float, exposure_fh: float, confidence: float = 0.90) -> float:
    """Exact coverage of the Garwood interval at a true rate, by summing the Poisson pmf."""
    mean = rate * exposure_fh / PER
    y_max = int(stats.poisson.ppf(1 - 1e-15, mean)) + 1
    total = 0.0
    for y in range(y_max + 1):
        lo, hi = garwood_interval(y, exposure_fh, confidence)
        if lo <= rate <= hi:
            total += float(stats.poisson.pmf(y, mean))
    return total


def expected_relative_width(rate: float, exposure_fh: float, confidence: float = 0.90) -> float:
    """Expected Garwood interval width divided by the true rate, over the Poisson count."""
    mean = rate * exposure_fh / PER
    y_max = int(stats.poisson.ppf(1 - 1e-15, mean)) + 1
    ys = np.arange(y_max + 1)
    widths = np.array(
        [np.subtract(*garwood_interval(int(y), exposure_fh, confidence)[::-1]) for y in ys]
    )
    return float(np.sum(stats.poisson.pmf(ys, mean) * widths) / rate)


def monthly_rates(counts: IntArray, exposure_fh: FloatArray) -> FloatArray:
    """Per-month rates per 1000 FH."""
    return PER * counts / exposure_fh
