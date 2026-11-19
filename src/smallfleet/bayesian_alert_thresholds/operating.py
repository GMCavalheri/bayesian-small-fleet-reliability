"""Operating characteristics of alert rules on simulated fleet histories.

Every replicate is a monthly history: ``baseline`` months at the reference rate, then
``monitor`` months either at the same rate (for false alarms) or at ``shift`` times it from
the first monitored month (for detection). Both the classical and the Bayesian rule see the
same histories, so their comparison is paired.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from smallfleet.bayesian_alert_thresholds.rules import window_exceedance
from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.data_generation import PER, FloatArray, IntArray
from smallfleet.frequentist_baseline import mean_sd_alert_level, monthly_rates

Rule = Callable[[IntArray, FloatArray], np.ndarray]
"""``(counts, exposure) -> boolean alert flag for every month`` of one history."""


@dataclass(frozen=True)
class Histories:
    exposure_fh: FloatArray
    """Monthly exposure, shared by every replicate."""
    steady: IntArray
    """Counts, shape (replicates, months), rate unchanged."""
    shifted: IntArray
    """Counts with the rate multiplied by ``shift`` from month ``baseline`` on."""
    baseline: int


def simulate_histories(
    rng: np.random.Generator,
    rate: float,
    month_fh: float,
    baseline: int,
    monitor: int,
    shift: float,
    replicates: int,
) -> Histories:
    months = baseline + monitor
    exposure = np.full(months, month_fh)
    m = rate * month_fh / PER
    steady = rng.poisson(m, (replicates, months))
    mult = np.where(np.arange(months) >= baseline, shift, 1.0)
    shifted = rng.poisson(m * mult, (replicates, months))
    return Histories(exposure, steady, shifted, baseline)


@dataclass(frozen=True)
class Characteristics:
    false_alarm_rate: float
    """Alerts per monitored month on unchanged histories."""
    detection: float
    """Share of shifted histories with at least one alert in the monitored months."""
    median_delay: float
    """Median months from the shift to the first alert, among detected histories (nan if none)."""


def characteristics(hist: Histories, rule: Rule) -> Characteristics:
    b = hist.baseline
    fa = np.mean([rule(h, hist.exposure_fh)[b:] for h in hist.steady])
    first = []
    for h in hist.shifted:
        flags = rule(h, hist.exposure_fh)[b:]
        hits = np.flatnonzero(flags)
        first.append(hits[0] + 1 if hits.size else -1)
    f = np.array(first)
    detected = f[f > 0]
    return Characteristics(
        false_alarm_rate=float(fa),
        detection=float(np.mean(f > 0)),
        median_delay=float(np.median(detected)) if detected.size else float("nan"),
    )


def classical_rule(k: float, baseline: int = 12) -> Rule:
    """``mean + k·sd`` of the first ``baseline`` monthly rates; flags every later month above."""

    def rule(counts: IntArray, exposure: FloatArray) -> np.ndarray:
        rates = monthly_rates(counts, exposure)
        level = mean_sd_alert_level(rates[:baseline], k)
        flags = rates > level
        flags[:baseline] = False
        return flags

    return rule


def bayes_window_rule(prior: GammaRate, target: float, p_star: float, window: int) -> Rule:
    """Alert when P(λ > target | prior, last ``window`` months) > ``p_star``."""

    def rule(counts: IntArray, exposure: FloatArray) -> np.ndarray:
        return window_exceedance(prior, counts, exposure, target, window) > p_star

    return rule
