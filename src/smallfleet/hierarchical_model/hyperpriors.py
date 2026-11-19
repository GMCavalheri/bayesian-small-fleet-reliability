"""Hyperpriors for the Gamma population of a hierarchical Poisson model.

The population of unit rates is Gamma(α, β), mean μ = α/β and coefficient of variation
cv = 1/sqrt(α). The exact oracle works on the grid ``u = log μ``, ``w = log α``, so every
hyperprior reports its log density in those coordinates, Jacobian included:

* a prior on (α, β): ``p(u, w) = p(α, β) · α · β``, because ``α = e^w``, ``β = e^(w − u)``;
* a prior on (μ, cv): ``p(u, w) = p(μ, cv) · μ · cv / 2``, because ``cv = e^(−w/2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from smallfleet.data_generation import FloatArray


@dataclass(frozen=True)
class AlphaBetaHyperprior:
    """Independent α ~ Exponential(mean ``alpha_mean``) and β ~ Gamma(``beta_shape``,
    ``beta_rate``): the hyperprior of NUREG/CR-6823 §8.3.4.2 with (1, 0.0625, 0.0625)."""

    alpha_mean: float = 1.0
    beta_shape: float = 0.0625
    beta_rate: float = 0.0625

    def log_density_uw(self, u: FloatArray, w: FloatArray) -> FloatArray:
        alpha = np.exp(w)
        beta = np.exp(w - u)
        lp = stats.expon.logpdf(alpha, scale=self.alpha_mean) + stats.gamma.logpdf(
            beta, self.beta_shape, scale=1.0 / self.beta_rate
        )
        return np.asarray(lp + np.log(alpha) + np.log(beta), dtype=np.float64)


@dataclass(frozen=True)
class MeanCVHyperprior:
    """μ ~ LogNormal(log ``mean_median``, ``mean_log_sd``), cv ~ HalfNormal(``cv_scale``).

    The project's default: the population mean is centred on a related large fleet's rate
    but allowed to differ by a factor of e^(±1.645·sd) (90 %), and the spread between units
    is weakly regularised towards zero (complete pooling) without forcing it.
    """

    mean_median: float
    mean_log_sd: float = 1.0
    cv_scale: float = 1.0

    def log_density_uw(self, u: FloatArray, w: FloatArray) -> FloatArray:
        mu = np.exp(u)
        cv = np.exp(-w / 2)
        lp = stats.lognorm.logpdf(
            mu, self.mean_log_sd, scale=self.mean_median
        ) + stats.halfnorm.logpdf(cv, scale=self.cv_scale)
        return np.asarray(lp + u + np.log(cv) - np.log(2.0), dtype=np.float64)
