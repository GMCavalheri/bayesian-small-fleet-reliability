"""PyMC against the exact oracle, posterior predictive checks and calibration."""

import numpy as np
import pytest
from scipy import stats

from smallfleet.hierarchical_model import (
    AlphaBetaHyperprior,
    MeanCVHyperprior,
    exact_posterior,
    pearson_dispersion,
    predictive_pvalue,
)
from smallfleet.hierarchical_model.model import build_model, sample

from nureg import example_8_1

NAMES, X, T = example_8_1()

# Sparse operators: 6 operators of 2-12 business jets, one year, starter-generator-like
# rates. Fixed data so the comparison is reproducible.
SPARSE_X = np.array([0, 1, 0, 2, 1, 4])
SPARSE_T = np.array([2, 3, 4, 6, 8, 12]) * 35 * 12 / 1000
SPARSE_PRIOR = MeanCVHyperprior(mean_median=0.2, mean_log_sd=1.0, cv_scale=1.0)

# MCMC agreement: |MCMC − exact| < 4 Monte Carlo standard errors (MCSE = sd/sqrt(ESS)).
# With a handful of compared quantities the chance of a false failure is below 1e-3.
Z = 4.0


def test_pearson_dispersion_reproduces_nureg_6823() -> None:
    # NUREG/CR-6823 §8.2.5.1: "The test statistic X2 was equal to 378.5" for Example 8.1,
    # compared with chi-squared on 65 degrees of freedom.
    assert pearson_dispersion(X, T) == pytest.approx(378.5, abs=0.05)
    assert stats.chi2.sf(378.5, 65) < 1e-40
    with pytest.raises(ValueError):
        pearson_dispersion([0, 0], [1.0, 1.0])


def test_predictive_check_rejects_pooling_and_accepts_the_population_model() -> None:
    rng = np.random.default_rng(3)
    post = exact_posterior(X, T, AlphaBetaHyperprior())
    a, b = post.sample_hyper(rng, 4000)
    p_hier = predictive_pvalue(rng, a, b, X, T)
    mu = X.sum() / T.sum()
    big = np.full(4000, 1e9)
    p_pool = predictive_pvalue(rng, big, big / mu, X, T)
    assert p_pool == 0.0  # X² = 378.5 is never reached by Poisson replicates
    assert 0.05 < p_hier < 0.95  # NUREG accepts the Gamma population for these data


def test_oracle_is_calibrated_by_simulation() -> None:
    # Simulation-based calibration (Cook, Gelman & Rubin 2006): draw (μ, cv) from the
    # hyperprior, rates and counts from the model, and compute the exact posterior CDF at the
    # true rate of unit 0. Those values are Uniform(0, 1), so the 90 % central interval must
    # cover the truth 90 % of the time. 300 replicates: standard error 0.017, 3 SE allowed.
    rng = np.random.default_rng(12)
    prior = MeanCVHyperprior(mean_median=0.2, mean_log_sd=0.5, cv_scale=0.5)
    t = SPARSE_T
    ranks = []
    for _ in range(300):
        mu = rng.lognormal(np.log(0.2), 0.5)
        cv = abs(rng.normal(0, 0.5))
        a = 1 / cv**2
        lam = rng.gamma(a, mu / a, t.size)
        x = rng.poisson(lam * t)
        post = exact_posterior(x, t, prior, n=121)
        ranks.append(post.unit_cdf(0, float(lam[0])))
    ranks_a = np.array(ranks)
    cover = np.mean((ranks_a > 0.05) & (ranks_a < 0.95))
    assert abs(cover - 0.90) < 3 * np.sqrt(0.09 / ranks_a.size)
    assert stats.kstest(ranks_a, "uniform").pvalue > 0.001


def test_grid_resolution_is_enough_for_sparse_data() -> None:
    # Means converge fastest. Tail quantiles carry the rectangle rule's error along the long
    # log-alpha direction (towards complete pooling); measured against n = 601 the error
    # oscillates at about 1e-4 relative for n = 241-361, and stays below 1e-3 at n = 121 (used
    # in the calibration study). Both are far below any Monte Carlo error compared with them.
    coarse = exact_posterior(SPARSE_X, SPARSE_T, SPARSE_PRIOR, n=121)
    default = exact_posterior(SPARSE_X, SPARSE_T, SPARSE_PRIOR)
    fine = exact_posterior(SPARSE_X, SPARSE_T, SPARSE_PRIOR, n=601)
    for i in range(SPARSE_X.size):
        assert coarse.unit_mean(i) == pytest.approx(fine.unit_mean(i), rel=1e-6)
        q = fine.unit_quantile(i, 0.95)
        assert default.unit_quantile(i, 0.95) == pytest.approx(q, rel=3e-4)
        assert coarse.unit_quantile(i, 0.95) == pytest.approx(q, rel=1e-3)


def test_both_model_forms_build() -> None:
    m1 = build_model(SPARSE_X, SPARSE_T, SPARSE_PRIOR, marginal=True)
    m2 = build_model(SPARSE_X, SPARSE_T, AlphaBetaHyperprior(), marginal=False)
    assert {"mu", "cv"} <= {v.name for v in m1.free_RVs}
    assert {"alpha", "beta", "lam"} <= {v.name for v in m2.free_RVs}


@pytest.mark.slow
def test_marginal_nuts_matches_the_oracle_on_nureg_data() -> None:
    r = sample(X, T, AlphaBetaHyperprior(), marginal=True, draws=1000, tune=1000, seed=7)
    exact = exact_posterior(X, T, AlphaBetaHyperprior())
    m = exact.hyper_moments()
    assert r.rhat_max < 1.01
    assert r.ess_bulk_min > 400 and r.ess_tail_min > 400
    assert r.divergences == 0
    assert abs(r.alpha.mean() - m["alpha_mean"]) < Z * m["alpha_sd"] / np.sqrt(r.ess_bulk_min)
    assert abs(r.beta.mean() - m["beta_mean"]) < Z * m["beta_sd"] / np.sqrt(r.ess_bulk_min)
    for plant in ("Callaway", "Oconee 2"):
        i = NAMES.index(plant)
        assert abs(r.unit_mean(i) - exact.unit_mean(i)) < Z * r.unit_mcse(i)


@pytest.mark.slow
def test_marginal_nuts_matches_the_oracle_on_sparse_operators() -> None:
    r = sample(SPARSE_X, SPARSE_T, SPARSE_PRIOR, marginal=True, draws=1000, tune=1000, seed=7)
    exact = exact_posterior(SPARSE_X, SPARSE_T, SPARSE_PRIOR)
    assert r.rhat_max < 1.01
    assert r.ess_bulk_min > 400
    assert r.divergences == 0
    for i in range(SPARSE_X.size):
        assert abs(r.unit_mean(i) - exact.unit_mean(i)) < Z * r.unit_mcse(i)
