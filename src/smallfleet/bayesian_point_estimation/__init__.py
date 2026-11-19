"""Phase 2: Bayesian estimation of a failure rate with the Gamma-Poisson conjugate model."""

from smallfleet.bayesian_point_estimation.conjugate import (
    GammaRate,
    constrained_noninformative,
    gamma_from_mean_cv,
    jeffreys,
)
from smallfleet.bayesian_point_estimation.evaluation import (
    Estimator,
    Performance,
    bayes_estimator,
    exact_performance,
    frequentist_estimator,
)
from smallfleet.bayesian_point_estimation.priors import (
    inflate_variance,
    power_prior,
    prior_predictive_pvalue,
)

__all__ = [
    "Estimator",
    "GammaRate",
    "Performance",
    "bayes_estimator",
    "constrained_noninformative",
    "exact_performance",
    "frequentist_estimator",
    "gamma_from_mean_cv",
    "inflate_variance",
    "jeffreys",
    "power_prior",
    "prior_predictive_pvalue",
]
