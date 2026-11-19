"""Priors borrowed from a large fleet, and a check that the borrowing is not contradicted.

A large fleet that flies the same component has logged ``Y`` events in ``T`` flight hours.
Taking its posterior as the small fleet's prior assumes the two fleets share one rate. They
rarely do (different sectors, cycles, climates), so the borrowed information is discounted:

* the **power prior** (Ibrahim & Chen 2000) raises the large-fleet likelihood to a power
  ``a0`` in [0, 1]. With a Jeffreys initial prior the result is Gamma(1/2 + a0·Y, a0·T): the
  large fleet counts as ``a0·T`` flight hours instead of ``T``. Here ``a0`` is set through
  that equivalent exposure, which is easier to reason about;
* **variance inflation** keeps the prior mean and multiplies its variance by ``k``, which
  divides both Gamma parameters by ``k``.

The prior-data conflict check (Box 1980) asks how surprising the small fleet's count is
under the prior predictive distribution.
"""

from __future__ import annotations

from smallfleet.bayesian_point_estimation.conjugate import GammaRate, jeffreys
from smallfleet.data_generation import PER


def power_prior(large_count: int, large_exposure_fh: float, equivalent_fh: float) -> GammaRate:
    """Jeffreys prior updated with the large fleet's data discounted to ``equivalent_fh``.

    ``a0 = equivalent_fh / large_exposure_fh``; ``equivalent_fh = 0`` gives the Jeffreys prior
    (improper) and ``equivalent_fh = large_exposure_fh`` the full large-fleet posterior.
    """
    if not 0 <= equivalent_fh <= large_exposure_fh or large_exposure_fh <= 0:
        raise ValueError("need 0 <= equivalent exposure <= large-fleet exposure")
    a0 = equivalent_fh / large_exposure_fh
    base = jeffreys()
    return GammaRate(base.alpha + a0 * large_count, a0 * large_exposure_fh / PER)


def inflate_variance(prior: GammaRate, factor: float) -> GammaRate:
    """Same mean, ``factor`` times the variance."""
    if factor < 1:
        raise ValueError("an inflation factor must be at least 1")
    return GammaRate(prior.alpha / factor, prior.beta / factor)


def prior_predictive_pvalue(prior: GammaRate, count: int, exposure_fh: float) -> float:
    """Two-sided tail probability of ``count`` under the prior predictive distribution.

    ``2·min(P(Y ≤ y), P(Y ≥ y))``, capped at 1. Small values flag prior-data conflict.
    """
    pred = prior.predictive(exposure_fh)
    lower = float(pred.cdf(count))
    upper = float(pred.sf(count - 1))
    return min(1.0, 2.0 * min(lower, upper))
