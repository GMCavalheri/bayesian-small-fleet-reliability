# `bayesian_point_estimation`: the Gamma-Poisson model and borrowed priors

**Lesson:** [2 · Borrowing a prior from a bigger fleet](lessons/02-borrowing-a-prior-from-a-bigger-fleet.md) ·
**Code:** [`src/smallfleet/bayesian_point_estimation/`](../src/smallfleet/bayesian_point_estimation/)

## Model

Gamma(α, β) prior (shape-rate, rate unit per 1000 FH, β in thousands of FH), Poisson data,
posterior Gamma(α + y, β + t). Predictive count: negative binomial with $n = \alpha$,
$p = \beta/(\beta + s)$. Priors: Jeffreys Gamma(½, 0); constrained noninformative
Gamma(½, 1/(2μ)); power prior Gamma(½ + a₀Y, a₀T) from a large fleet's $Y$ events in $T$;
variance inflation Gamma(α/k, β/k). Prior-data conflict: two-sided prior-predictive tail.

## API

| function | purpose |
|---|---|
| `GammaRate` (`mean`, `var`, `ppf`, `cdf`, `sf`, `interval`, `hpd`, `update`, `predictive`) | the distribution |
| `jeffreys`, `constrained_noninformative`, `gamma_from_mean_cv` | priors |
| `power_prior`, `inflate_variance`, `prior_predictive_pvalue` | borrowing |
| `exact_performance`, `frequentist_estimator`, `bayes_estimator` | exact bias, RMSE, coverage, width |

## Verification

All four rows of NUREG/CR-6823 Table 6.4; posterior and predictive against numerical
integration and Monte Carlo; HPD properties; power-prior limits; posterior mean between prior
mean and MLE (Hypothesis)
([`tests/test_bayesian_point_estimation.py`](../tests/test_bayesian_point_estimation.py)).
