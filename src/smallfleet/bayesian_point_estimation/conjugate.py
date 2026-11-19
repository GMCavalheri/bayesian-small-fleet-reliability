"""The Gamma-Poisson conjugate model.

A Gamma(α, β) prior on a rate λ (shape-rate form, mean α/β) and a Poisson count
``y ~ Poisson(λ t)`` give the posterior Gamma(α + y, β + t), where ``t`` is the exposure in
the rate's unit (thousands of flight hours here). The prior is "worth" α events in β units of
exposure (NUREG/CR-6823 §6.2.2.4). The predictive count for a future exposure ``s`` is
negative binomial with ``n = α`` and ``p = β / (β + s)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize, stats

from smallfleet.data_generation import PER


@dataclass(frozen=True)
class GammaRate:
    """A Gamma(alpha, beta) distribution for a rate per 1000 FH (beta in thousands of FH).

    ``beta = 0`` is allowed for the improper Jeffreys prior Gamma(1/2, 0); such a
    distribution can be updated but has no moments or quantiles.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta < 0:
            raise ValueError("need alpha > 0 and beta >= 0")

    @property
    def proper(self) -> bool:
        return self.beta > 0

    def _scale(self) -> float:
        if not self.proper:
            raise ValueError("an improper Gamma(alpha, 0) has no distribution")
        return 1.0 / self.beta

    def mean(self) -> float:
        return self.alpha * self._scale()

    def var(self) -> float:
        return self.alpha * self._scale() ** 2

    def sd(self) -> float:
        return math.sqrt(self.var())

    def ppf(self, q: float) -> float:
        return float(stats.gamma.ppf(q, self.alpha, scale=self._scale()))

    def cdf(self, x: float) -> float:
        return float(stats.gamma.cdf(x, self.alpha, scale=self._scale()))

    def sf(self, x: float) -> float:
        """``P(λ > x)``."""
        return float(stats.gamma.sf(x, self.alpha, scale=self._scale()))

    def pdf(self, x: float) -> float:
        return float(stats.gamma.pdf(x, self.alpha, scale=self._scale()))

    def interval(self, mass: float = 0.90) -> tuple[float, float]:
        """Equal-tailed credible interval."""
        a = (1 - mass) / 2
        return self.ppf(a), self.ppf(1 - a)

    def hpd(self, mass: float = 0.90) -> tuple[float, float]:
        """Highest-posterior-density interval: the shortest interval holding ``mass``.

        For ``alpha <= 1`` the density is decreasing and the interval starts at 0.
        """
        if self.alpha <= 1:
            return 0.0, self.ppf(mass)

        def width(lower_tail: float) -> float:
            return self.ppf(lower_tail + mass) - self.ppf(lower_tail)

        res = optimize.minimize_scalar(
            width, bounds=(0.0, 1.0 - mass), method="bounded", options={"xatol": 1e-12}
        )
        a = float(res.x)
        return self.ppf(a), self.ppf(a + mass)

    def update(self, count: int, exposure_fh: float) -> GammaRate:
        """Posterior after ``count`` events in ``exposure_fh`` flight hours."""
        if count < 0 or exposure_fh < 0:
            raise ValueError("count and exposure must be non-negative")
        return GammaRate(self.alpha + count, self.beta + exposure_fh / PER)

    def predictive(self, exposure_fh: float) -> NegativeBinomial:
        """Negative-binomial distribution of the count in a future ``exposure_fh``."""
        self._scale()
        s = exposure_fh / PER
        return NegativeBinomial(self.alpha, self.beta / (self.beta + s))


@dataclass(frozen=True)
class NegativeBinomial:
    """``P(Y = y) = Γ(n + y) / (Γ(n) y!) · p^n (1 − p)^y``, the Gamma-Poisson mixture."""

    n: float
    p: float

    def pmf(self, y: int) -> float:
        return float(stats.nbinom.pmf(y, self.n, self.p))

    def cdf(self, y: int) -> float:
        return float(stats.nbinom.cdf(y, self.n, self.p))

    def sf(self, y: int) -> float:
        """``P(Y > y)``."""
        return float(stats.nbinom.sf(y, self.n, self.p))

    def mean(self) -> float:
        return self.n * (1 - self.p) / self.p

    def var(self) -> float:
        return self.n * (1 - self.p) / self.p**2


def jeffreys() -> GammaRate:
    """The Jeffreys noninformative prior for a Poisson rate, Gamma(1/2, 0) (improper)."""
    return GammaRate(0.5, 0.0)


def constrained_noninformative(mean: float) -> GammaRate:
    """Gamma(1/2, 1/(2·mean)): the most diffuse prior with a given mean (Atwood 1996)."""
    if mean <= 0:
        raise ValueError("mean must be positive")
    return GammaRate(0.5, 0.5 / mean)


def gamma_from_mean_cv(mean: float, cv: float) -> GammaRate:
    """Moment-matched Gamma: ``alpha = 1/cv²``, ``beta = alpha/mean``."""
    if mean <= 0 or cv <= 0:
        raise ValueError("mean and cv must be positive")
    alpha = 1.0 / cv**2
    return GammaRate(alpha, alpha / mean)
