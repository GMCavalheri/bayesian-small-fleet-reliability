# `empirical_bayes`: estimating the population from many units

**Lesson:** [3 · Letting the data choose the prior](lessons/03-letting-the-data-choose-the-prior.md) ·
**Code:** [`src/smallfleet/empirical_bayes/`](../src/smallfleet/empirical_bayes/)

## Model

$x_i \sim \text{Poisson}(\lambda_i t_i)$, $\lambda_i \sim \text{Gamma}(\alpha, \beta)$; marginal
$x_i \sim \text{NB}(\alpha, \beta/(\beta + t_i))$. (α, β) by maximum marginal likelihood
(bounded L-BFGS-B on (log μ, log α) with the analytic score) or by moments
($\hat\sigma^2 = \max(0, \sum[(x_i - \hat\mu t_i)^2 - \hat\mu t_i]/\sum t_i^2)$). Plug-in posteriors
Gamma(α̂ + x_i, β̂ + t_i); Kass-Steffey variance adjustment, moment-matched to a Gamma.

## API

| function | purpose |
|---|---|
| `marginal_loglik`, `fit_marginal_mle`, `method_of_moments` | the population |
| `plugin_posteriors`, `kass_steffey_posteriors` | unit posteriors |
| `EmpiricalBayesFit` (`alpha`, `beta`, `pooled`, `prior`) | the fit; `pooled` at zero spread |

## Failure modes

No events in any unit: no maximum (ValueError). Equidispersed data: maximum on the
zero-spread boundary (`pooled=True`). Non-positive information for α: Kass-Steffey is
undefined (ValueError).

## Verification

NUREG/CR-6823 Tables 8.2 and 8.3 on Example 8.1's 66 plants (the printed ∂E/∂α̂ is
inconsistent with Table 8.3; ours matches); MLE against an independent optimiser; score
against finite differences; recovery of known hyperparameters
([`tests/test_empirical_bayes.py`](../tests/test_empirical_bayes.py)).
