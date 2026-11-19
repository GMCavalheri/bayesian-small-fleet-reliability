"""Empirical Bayes: let several fleets' data choose the prior.

``m`` units (fleets, operators, plants) each log ``x_i`` events in exposure ``t_i`` (thousands
of FH). Each unit has its own rate, drawn from one population Gamma(α, β). Marginally,
``x_i`` is negative binomial, so α and β can be estimated from all units at once, by maximum
marginal likelihood or by moments. Each unit's posterior is then Gamma(α̂ + x_i, β̂ + t_i).

Plugging in α̂ and β̂ ignores their uncertainty, so the intervals are too narrow. The
Kass-Steffey (1989) adjustment widens each posterior variance by a delta-method term, in
the parameterisation μ = α/β, α where the two estimates are asymptotically uncorrelated
(NUREG/CR-6823 §8.2.4.1).
"""

from smallfleet.empirical_bayes.estimation import (
    EmpiricalBayesFit,
    fit_marginal_mle,
    kass_steffey_posteriors,
    marginal_loglik,
    method_of_moments,
    plugin_posteriors,
)

__all__ = [
    "EmpiricalBayesFit",
    "fit_marginal_mle",
    "kass_steffey_posteriors",
    "marginal_loglik",
    "method_of_moments",
    "plugin_posteriors",
]
