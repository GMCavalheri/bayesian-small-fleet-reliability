"""Alerts stated as posterior probabilities.

The classical alert asks "is this month's rate above a line drawn from last year's scatter?".
The Bayesian alert asks the question an engineer means: "how probable is it that the true
rate now exceeds the rate we can accept?"

```
alert in month j  ⇔  P(λ > λ_target | prior, data of the last w months) > p*
```

With a Gamma prior and Poisson counts the probability is a Gamma survival function, so the
rule is equivalent to a threshold on the window's event count: alert when the count reaches
the smallest ``c`` for which the exceedance probability passes ``p*``. That makes the
false-alarm rate at a constant true rate an exact Poisson tail.

A geometric forgetting factor ``δ`` (data from ``k`` months ago weighted by ``δ^k``) is the
smooth alternative to a hard window; it has no exact false-alarm formula.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.data_generation import PER, FloatArray, IntArray


def exceedance_probability(
    prior: GammaRate, count: int, exposure_fh: float, target: float
) -> float:
    """``P(λ > target | count in exposure)`` under the conjugate update of ``prior``."""
    return prior.update(count, exposure_fh).sf(target)


def alert_count_threshold(
    prior: GammaRate, exposure_fh: float, target: float, p_star: float, max_count: int = 100_000
) -> int:
    """Smallest count ``c`` in ``exposure_fh`` with ``P(λ > target | c) > p_star``.

    The exceedance probability rises with the count (the posterior is stochastically
    increasing), so a bisection over integers finds it.
    """
    if not 0 < p_star < 1 or target <= 0:
        raise ValueError("need 0 < p_star < 1 and a positive target")
    if exceedance_probability(prior, max_count, exposure_fh, target) <= p_star:
        raise ValueError("no count up to max_count reaches p_star")
    lo, hi = -1, max_count  # invariant: lo fails (or is -1), hi passes
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if exceedance_probability(prior, mid, exposure_fh, target) > p_star:
            hi = mid
        else:
            lo = mid
    return hi


def window_exceedance(
    prior: GammaRate, counts: IntArray, exposure_fh: FloatArray, target: float, window: int
) -> FloatArray:
    """Exceedance probability in every month from the prior and the last ``window`` months.

    Months before the history starts count as missing (shorter windows early on).
    """
    c = np.concatenate([[0], np.cumsum(counts)])
    e = np.concatenate([[0.0], np.cumsum(exposure_fh)])
    j = np.arange(1, counts.size + 1)
    start = np.maximum(j - window, 0)
    a = prior.alpha + (c[j] - c[start])
    b = prior.beta + (e[j] - e[start]) / PER
    return np.asarray(special.gammaincc(a, target * b), dtype=np.float64)


def discounted_exceedance(
    prior: GammaRate, counts: IntArray, exposure_fh: FloatArray, target: float, delta: float
) -> FloatArray:
    """Exceedance probability with past months weighted by ``δ^age`` (``δ = 1``: no forgetting)."""
    if not 0 < delta <= 1:
        raise ValueError("delta must be in (0, 1]")
    a = np.empty(counts.size)
    b = np.empty(counts.size)
    sa = sb = 0.0
    for k in range(counts.size):
        sa = delta * sa + counts[k]
        sb = delta * sb + exposure_fh[k] / PER
        a[k], b[k] = prior.alpha + sa, prior.beta + sb
    return np.asarray(special.gammaincc(a, target * b), dtype=np.float64)


def window_false_alarm_rate(
    prior: GammaRate,
    rate: float,
    month_fh: float,
    window: int,
    target: float,
    p_star: float,
) -> float:
    """Exact per-month alert probability of the full-window rule at a constant true rate.

    The window holds ``window`` months of exposure; its count is Poisson, and the rule alerts
    when the count reaches :func:`alert_count_threshold`.
    """
    exposure = window * month_fh
    c = alert_count_threshold(prior, exposure, target, p_star)
    return float(stats.poisson.sf(c - 1, rate * exposure / PER))
