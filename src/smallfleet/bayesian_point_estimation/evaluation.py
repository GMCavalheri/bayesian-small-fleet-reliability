"""Exact repeated-sampling performance of a rate estimator.

For a true rate λ and exposure T, the count is Poisson(λT/1000), so any estimator that maps a
count to a point estimate and an interval has an exact bias, root-mean-square error,
coverage and expected interval width: a sum over counts weighted by the Poisson pmf. No
simulation noise, so results reproduce byte for byte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from smallfleet.bayesian_point_estimation.conjugate import GammaRate
from smallfleet.data_generation import PER
from smallfleet.frequentist_baseline import garwood_interval, mle_rate

Estimator = Callable[[int, float], tuple[float, float, float]]
"""``(count, exposure_fh) -> (point estimate, interval lower, interval upper)``."""


@dataclass(frozen=True)
class Performance:
    bias: float
    rmse: float
    coverage: float
    mean_width: float


def exact_performance(estimator: Estimator, rate: float, exposure_fh: float) -> Performance:
    """Bias, RMSE, coverage and mean interval width, summed over the Poisson count."""
    m = rate * exposure_fh / PER
    y_max = int(stats.poisson.ppf(1 - 1e-14, m)) + 2
    ys = np.arange(y_max + 1)
    w = stats.poisson.pmf(ys, m)
    w = w / w.sum()
    est = np.array([estimator(int(y), exposure_fh) for y in ys])
    point, lo, hi = est[:, 0], est[:, 1], est[:, 2]
    err = point - rate
    return Performance(
        bias=float(w @ err),
        rmse=float(np.sqrt(w @ err**2)),
        coverage=float(w @ ((lo <= rate) & (rate <= hi))),
        mean_width=float(w @ (hi - lo)),
    )


def frequentist_estimator(mass: float = 0.90) -> Estimator:
    """MLE with the Garwood interval."""

    def est(y: int, t_fh: float) -> tuple[float, float, float]:
        return (mle_rate(y, t_fh), *garwood_interval(y, t_fh, mass))

    return est


def bayes_estimator(prior: GammaRate, mass: float = 0.90) -> Estimator:
    """Posterior mean with the equal-tailed credible interval."""

    def est(y: int, t_fh: float) -> tuple[float, float, float]:
        post = prior.update(y, t_fh)
        return (post.mean(), *post.interval(mass))

    return est
