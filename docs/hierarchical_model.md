# `hierarchical_model`: exact and MCMC partial pooling

**Lesson:** [4 · Partial pooling](lessons/04-partial-pooling.md) ·
**Code:** [`src/smallfleet/hierarchical_model/`](../src/smallfleet/hierarchical_model/)

## Model

The empirical Bayes model with a hyperprior: (α, β) with α ~ Exponential and β ~ Gamma
(NUREG/CR-6823 §8.3.4), or (μ, cv) with μ ~ LogNormal and cv ~ HalfNormal.

* **Oracle.** The posterior of (log μ, log α) on a grid: a wide scan of log α over [−12, 46],
  refinement where the density is within $e^{-30}$ of its peak, and trimming to all but 1e−13
  of the mass. Unit posteriors are exact Gamma mixtures. The log-likelihood is stable to
  α ~ 1e20 ($\sum_{k<x}\ln(\alpha + k)$ and log1p).
* **PyMC.** The marginal form (NB likelihood, rates drawn from their exact conditionals) or the
  centred form; NUTS on one core with a fixed seed; R-hat, bulk/tail ESS, divergences.
* **Checks.** Pearson dispersion and its posterior predictive p-value.

## API

| function | purpose |
|---|---|
| `AlphaBetaHyperprior`, `MeanCVHyperprior` | hyperpriors with Jacobians |
| `exact_posterior(x, t, hyperprior, n)` → `HierarchicalPosterior` | the oracle |
| `.unit_mean/var/cdf/quantile/interval/sf`, `.hyper_moments`, `.hyper_quantile`, `.sample_hyper` | posterior summaries |
| `model.build_model`, `model.sample` → `McmcResult` | PyMC |
| `pearson_dispersion`, `predictive_pvalue` | model checks |

## Verification

NUREG/CR-6823 Table 8.7 (WinBUGS) and §8.3.4.2 moments; independent `dblquad` integration;
grid refinement; Jacobians integrate to 1; simulation-based calibration; NUTS within 4 MCSE
of the oracle with R-hat < 1.01 and no divergences; X² = 378.5 for Example 8.1
([`tests/test_hierarchical_oracle.py`](../tests/test_hierarchical_oracle.py),
[`tests/test_hierarchical_model.py`](../tests/test_hierarchical_model.py)).
