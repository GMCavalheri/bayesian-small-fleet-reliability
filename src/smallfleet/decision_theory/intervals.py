"""A failure-finding interval chosen under rate uncertainty.

A protective device fails at constant rate λ, and the failure stays hidden until the next
test, every τ flight hours. Averaged over an interval, the fraction of time it is failed and
undetected (its mean unavailability) is

```
U(λ, τ) = 1 − (1 − e^(−λτ)) / (λτ)  ≈  λτ/2   (λτ small),
```

which gives the classical failure-finding interval τ ≈ 2·U*/λ for a target U* (Moubray 1997).
With λ uncertain, two defensible rules replace the point estimate:

* **expected unavailability:** choose the longest τ with E[U(λ, τ)] ≤ U*. For a
  Gamma(a, b) posterior (a > 1) the expectation is closed-form:
  ``E[U] = 1 − b/((a − 1)τ) · (1 − (b/(b + τ))^(a−1))``;
* **risk-based:** choose the longest τ with P(U(λ, τ) > U*) ≤ 1 − q. U rises with λ, so this
  is the known-rate interval at the posterior q-quantile of λ.

Rates are per 1000 FH and τ is in FH throughout.
"""

from __future__ import annotations

import math

from scipy import optimize

from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.data_generation import PER


def unavailability(rate: float, tau_fh: float) -> float:
    """Mean unavailability over a test interval, ``1 − (1 − e^(−x))/x`` with ``x = λτ``."""
    x = rate * tau_fh / PER
    if x < 1e-8:
        return x / 2 - x * x / 6
    return 1.0 - (-math.expm1(-x)) / x


def interval_for_known_rate(rate: float, target: float) -> float:
    """Longest τ with ``U(λ, τ) ≤ target`` (U rises with τ)."""
    if not 0 < target < 1 or rate <= 0:
        raise ValueError("need a target in (0, 1) and a positive rate")
    hi = 2 * target / rate * PER
    while unavailability(rate, hi) < target:
        hi *= 2
    return float(optimize.brentq(lambda t: unavailability(rate, t) - target, 0.0, hi, xtol=1e-10))


def expected_unavailability(posterior: GammaRate, tau_fh: float) -> float:
    """``E[U(λ, τ)]`` under a Gamma(a, b) posterior, closed form (requires a > 1)."""
    a, b = posterior.alpha, posterior.beta
    if a <= 1:
        raise ValueError("the closed form needs alpha > 1 (E[1/λ] must exist)")
    t = tau_fh / PER
    return 1.0 - b / ((a - 1) * t) * (-math.expm1((a - 1) * math.log(b / (b + t))))


def interval_for_expected_unavailability(posterior: GammaRate, target: float) -> float:
    """Longest τ with ``E[U(λ, τ)] ≤ target``."""
    hi = 2 * target / posterior.mean() * PER
    while expected_unavailability(posterior, hi) < target:
        hi *= 2
    return float(
        optimize.brentq(
            lambda t: expected_unavailability(posterior, t) - target, 1e-9, hi, xtol=1e-10
        )
    )


def interval_for_risk(posterior: GammaRate, target: float, q: float) -> float:
    """Longest τ with ``P(U(λ, τ) > target) ≤ 1 − q``: the interval at the q-quantile of λ."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    return interval_for_known_rate(posterior.ppf(q), target)
