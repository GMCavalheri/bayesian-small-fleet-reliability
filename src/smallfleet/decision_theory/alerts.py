"""Investigate or wait: the alert as an expected-loss decision.

States: the component's rate is above target (λ > λ*) or not. Actions: investigate now, or
not. Losses: an investigation that finds nothing costs ``c_FA``; not investigating when the
rate is above target costs ``c_miss`` (per decision period). With ``p = P(λ > λ* | data)``

```
E[loss | investigate] = c_FA (1 − p),      E[loss | wait] = c_miss p,
```

so investigating is optimal exactly when ``p > c_FA / (c_FA + c_miss)``: the Bayesian alert
threshold of Lesson 5 is a ratio of costs, not a convention.

Waiting a month buys data. Its expected value (the expected value of sample information,
EVSI) is the expected reduction in the minimum expected loss after seeing next month's
count, averaged over the predictive distribution of that count. EVSI is never negative.
Waiting is worth it when EVSI exceeds the cost of delay, ``c_delay · p``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.data_generation import PER


@dataclass(frozen=True)
class AlertCosts:
    false_alarm: float
    missed: float
    delay: float = 0.0

    def __post_init__(self) -> None:
        if self.false_alarm <= 0 or self.missed <= 0 or self.delay < 0:
            raise ValueError("costs must be positive (delay non-negative)")


def bayes_threshold(costs: AlertCosts) -> float:
    """The exceedance probability above which investigating has the lower expected loss."""
    return costs.false_alarm / (costs.false_alarm + costs.missed)


def expected_losses(p_exceed: float, costs: AlertCosts) -> tuple[float, float]:
    """(E[loss | investigate], E[loss | wait]) for exceedance probability ``p_exceed``."""
    return costs.false_alarm * (1 - p_exceed), costs.missed * p_exceed


def _min_loss(p: np.ndarray, costs: AlertCosts) -> np.ndarray:
    return np.minimum(costs.false_alarm * (1 - p), costs.missed * p)


@dataclass(frozen=True)
class WaitValue:
    loss_now: float
    """Minimum expected loss of deciding now."""
    loss_after: float
    """Expected minimum loss after one more month of data (before the delay cost)."""
    evsi: float
    """``loss_now − loss_after`` ≥ 0."""
    delay_cost: float
    """``c_delay · P(λ > λ*)``."""

    @property
    def wait_is_better(self) -> bool:
        return self.evsi > self.delay_cost


def value_of_waiting(
    posterior: GammaRate, target: float, month_fh: float, costs: AlertCosts
) -> WaitValue:
    """Preposterior analysis of waiting one month before deciding.

    Next month's count is negative binomial under ``posterior``; for each possible count the
    updated exceedance probability gives the optimal action's loss.
    """
    p_now = posterior.sf(target)
    loss_now = float(_min_loss(np.array(p_now), costs))
    pred = posterior.predictive(month_fh)
    s = month_fh / PER
    y_max = int(stats.nbinom.ppf(1 - 1e-14, pred.n, pred.p)) + 2
    ys = np.arange(y_max + 1)
    w = stats.nbinom.pmf(ys, pred.n, pred.p)
    p_next = special.gammaincc(posterior.alpha + ys, target * (posterior.beta + s))
    loss_after = float(np.sum(w * _min_loss(p_next, costs)) / w.sum())
    return WaitValue(
        loss_now=loss_now,
        loss_after=loss_after,
        # EVSI ≥ 0 exactly (the minimum of expectations is at least the expected minimum);
        # the guard only removes a negative rounding residue of order 1e-15.
        evsi=max(loss_now - loss_after, 0.0),
        delay_cost=costs.delay * p_now,
    )
