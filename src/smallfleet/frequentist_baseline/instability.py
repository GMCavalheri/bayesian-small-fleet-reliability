"""Why the classical alert level degrades as a fleet gets smaller.

With a Poisson mean ``m = λT/1000`` per month, two things happen as ``m`` falls: most months
log zero events, and a 12-month baseline is often all zeros, so its standard deviation is 0
and the alert level collapses to the mean (any single event alerts). Both have closed forms.
The operating characteristics of the alert rule are estimated by seeded Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from smallfleet.data_generation import PER
from smallfleet.frequentist_baseline.alert_level import mean_sd_alert_level


def zero_month_probability(rate: float, exposure_fh: float) -> float:
    """``P(Y = 0) = exp(−m)`` for one month."""
    return float(np.exp(-rate * exposure_fh / PER))


def degenerate_baseline_probability(rate: float, exposure_fh: float, months: int = 12) -> float:
    """Probability that every baseline month logs the same count, so the sd is zero.

    ``Σ_k P(Y = k)^months`` over all counts ``k``.
    """
    mean = rate * exposure_fh / PER
    k_max = int(stats.poisson.ppf(1 - 1e-15, mean)) + 1
    pmf = stats.poisson.pmf(np.arange(k_max + 1), mean)
    return float(np.sum(pmf**months))


@dataclass(frozen=True)
class OperatingCharacteristics:
    false_alarm_rate: float
    """Share of monitored months that alert while the true rate is unchanged."""
    false_alarm_se: float
    detection_probability: float
    """Share of replicates that alert at least once in the monitored window after the rate
    rises by ``shift`` from the first monitored month."""
    detection_se: float
    degenerate_share: float
    """Share of baselines with zero standard deviation."""


def mean_sd_operating_characteristics(
    rng: np.random.Generator,
    rate: float,
    exposure_fh: float,
    k: float = 2.0,
    shift: float = 2.0,
    baseline_months: int = 12,
    monitor_months: int = 3,
    replicates: int = 20_000,
) -> OperatingCharacteristics:
    """Monte Carlo operating characteristics of the ``mean + k·sd`` rule on monthly rates."""
    m = rate * exposure_fh / PER
    base = rng.poisson(m, (replicates, baseline_months)) * PER / exposure_fh
    levels = np.array([mean_sd_alert_level(b, k) for b in base])
    degenerate = float(np.mean(np.std(base, axis=1) == 0))
    same = rng.poisson(m, (replicates, monitor_months)) * PER / exposure_fh
    fa = float(np.mean(same > levels[:, None]))
    worse = rng.poisson(m * shift, (replicates, monitor_months)) * PER / exposure_fh
    det = float(np.mean(np.any(worse > levels[:, None], axis=1)))
    n_fa = replicates * monitor_months
    return OperatingCharacteristics(
        false_alarm_rate=fa,
        false_alarm_se=float(np.sqrt(fa * (1 - fa) / n_fa)),
        detection_probability=det,
        detection_se=float(np.sqrt(det * (1 - det) / replicates)),
        degenerate_share=degenerate,
    )
