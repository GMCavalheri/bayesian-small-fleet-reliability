import numpy as np
import pytest
from scipy import optimize

from smallfleet.empirical_bayes import (
    fit_marginal_mle,
    kass_steffey_posteriors,
    marginal_loglik,
    method_of_moments,
    plugin_posteriors,
)

from nureg import example_8_1

NAMES, X, T = example_8_1()

# NUREG/CR-6823 Tables 8.2 (plug-in) and 8.3 (Kass-Steffey): Gamma parameters and
# (5th percentile, mean, 95th percentile). Plant names as in Example 8.1's data table.
TABLE_8_2 = {
    "Callaway": (13.39, 2.715, (2.94, 4.93, 7.34)),
    "Wash. Nucl. 2": (24.39, 5.575, (3.03, 4.37, 5.93)),
    "Salem 1": (11.39, 3.885, (1.66, 2.93, 4.49)),
    "Diablo Canyon 1": (6.39, 2.296, (1.25, 2.78, 4.81)),
    "Grand Gulf": (8.39, 3.301, (1.29, 2.54, 4.14)),
    "Pt. Beach 2": (1.39, 8.755, (0.0164, 0.159, 0.424)),
    "Prairie Island 2": (1.39, 9.055, (0.0158, 0.154, 0.410)),
    "Oconee 2": (1.39, 9.995, (0.0143, 0.139, 0.372)),
}
TABLE_8_3 = {
    "Callaway": (12.13, 2.460, (2.86, 4.93, 7.47)),
    "Wash. Nucl. 2": (23.40, 5.348, (3.00, 4.37, 5.96)),
    "Salem 1": (11.03, 3.762, (1.65, 2.93, 4.52)),
    "Diablo Canyon 1": (6.08, 2.185, (1.22, 2.78, 4.86)),
    "Grand Gulf": (8.15, 3.204, (1.27, 2.54, 4.16)),
    "Pt. Beach 2": (1.33, 8.382, (0.0151, 0.159, 0.431)),
    "Prairie Island 2": (1.33, 8.665, (0.0146, 0.154, 0.417)),
    "Oconee 2": (1.33, 9.554, (0.0132, 0.139, 0.378)),
}


def decimals(printed: float) -> int:
    text = f"{printed}"
    return len(text.split(".")[1]) if "." in text else 0


def assert_printed(value: float, printed: float, units: float = 0.5) -> None:
    # A printed value is matched to within `units` of its last printed digit.
    step = 10.0 ** -decimals(printed)
    assert abs(value - printed) <= units * step + 1e-12, (value, printed)


def test_fixture_totals_match_table_8_1() -> None:
    # Table 8.1, "Industry": 361 events in 374.229 thousand critical hours, 66 plants.
    assert len(NAMES) == 66
    assert X.sum() == 361
    assert T.sum() == pytest.approx(374.229, abs=5e-4)


def test_marginal_mle_reproduces_nureg_6823_hyperparameters() -> None:
    # Table 8.2, "Industry": the fitted population distribution is Gamma(1.39, 1.211), with
    # 90 % interval (0.118, 1.15, 3.07).
    fit = fit_marginal_mle(X, T)
    assert not fit.pooled
    assert_printed(fit.alpha, 1.39)
    assert_printed(fit.beta, 1.211)
    prior = fit.prior
    assert_printed(prior.ppf(0.05), 0.118)
    assert_printed(prior.mean(), 1.15)
    assert_printed(prior.ppf(0.95), 3.07)


def test_marginal_mle_is_the_maximum_found_by_brute_force() -> None:
    # Independent oracle: a different optimiser (L-BFGS-B on (alpha, beta) directly).
    fit = fit_marginal_mle(X, T)
    res = optimize.minimize(
        lambda p: -marginal_loglik(p[0], p[1], X, T),
        x0=np.array([1.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(1e-3, 100.0), (1e-3, 100.0)],
    )
    assert fit.loglik >= -res.fun - 1e-8
    assert fit.alpha == pytest.approx(res.x[0], rel=1e-4)
    assert fit.beta == pytest.approx(res.x[1], rel=1e-4)


@pytest.mark.parametrize("plant", sorted(TABLE_8_2))
def test_plugin_posteriors_reproduce_table_8_2(plant: str) -> None:
    alpha, beta, (lo, mean, hi) = TABLE_8_2[plant]
    post = plugin_posteriors(fit_marginal_mle(X, T), X, T)[NAMES.index(plant)]
    assert_printed(post.alpha, alpha)
    assert_printed(post.beta, beta)
    assert_printed(post.ppf(0.05), lo)
    assert_printed(post.mean(), mean)
    assert_printed(post.ppf(0.95), hi)


@pytest.mark.parametrize("plant", sorted(TABLE_8_3))
def test_kass_steffey_posteriors_reproduce_table_8_3(plant: str) -> None:
    # The Gamma parameters are matched to one unit in the last printed digit, the
    # percentiles and mean to half a unit. Reason: the handbook works from its own α̂, of
    # which it prints only 1.39; our α̂ = 1.3905. Wash. Nucl. 2's adjusted α is 23.3945,
    # 0.0005 under the 23.395 rounding boundary, so a 1e-4 change in α̂ flips its last digit.
    alpha, beta, (lo, mean, hi) = TABLE_8_3[plant]
    post = kass_steffey_posteriors(fit_marginal_mle(X, T), X, T)[NAMES.index(plant)]
    assert_printed(post.alpha, alpha, units=1.0)
    assert_printed(post.beta, beta, units=1.0)
    assert_printed(post.ppf(0.05), lo)
    assert_printed(post.mean(), mean)
    assert_printed(post.ppf(0.95), hi)


def test_kass_steffey_keeps_the_mean_and_widens_the_variance() -> None:
    fit = fit_marginal_mle(X, T)
    for plug, ks in zip(
        plugin_posteriors(fit, X, T), kass_steffey_posteriors(fit, X, T), strict=True
    ):
        assert ks.mean() == pytest.approx(plug.mean(), rel=1e-12)
        assert ks.var() >= plug.var()


def test_estimators_recover_known_hyperparameters() -> None:
    # Monte Carlo: 400 units drawn from Gamma(2, 10) (mean 0.2, cv 0.71), exposures 5-50.
    # Over 50 seeded replicates the mean estimates must sit within 4 standard errors of the
    # truth for μ; for α (biased in finite samples) within 15 % of the truth.
    rng = np.random.default_rng(8)
    mle_mu, mle_a, mom_mu, mom_a = [], [], [], []
    for _ in range(50):
        t = rng.uniform(5, 50, 400)
        lam = rng.gamma(2.0, 1 / 10.0, 400)
        x = rng.poisson(lam * t)
        f, m = fit_marginal_mle(x, t), method_of_moments(x, t)
        mle_mu.append(f.mean)
        mle_a.append(f.alpha)
        mom_mu.append(m.mean)
        mom_a.append(m.alpha)
    for mus in (mle_mu, mom_mu):
        assert abs(np.mean(mus) - 0.2) < 4 * np.std(mus) / np.sqrt(len(mus))
    assert np.mean(mle_a) == pytest.approx(2.0, rel=0.15)
    assert np.mean(mom_a) == pytest.approx(2.0, rel=0.15)


def test_no_spread_returns_the_pooled_fit() -> None:
    t = np.full(20, 10.0)
    x = np.full(20, 2)  # under-dispersed: every unit identical
    fit = fit_marginal_mle(x, t)
    assert fit.pooled and fit.mean == pytest.approx(0.2)
    mom = method_of_moments(x, t)
    assert mom.pooled
    with pytest.raises(ValueError):
        kass_steffey_posteriors(fit, x, t)


def test_no_events_has_no_estimate() -> None:
    t = np.full(5, 3.0)
    with pytest.raises(ValueError, match="no events"):
        fit_marginal_mle(np.zeros(5, dtype=int), t)
    with pytest.raises(ValueError, match="no events"):
        method_of_moments(np.zeros(5, dtype=int), t)


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        fit_marginal_mle([1], [1.0])
    with pytest.raises(ValueError):
        method_of_moments([1, -1], [1.0, 1.0])


def test_kass_steffey_refuses_a_non_positive_information() -> None:
    # A real case from the operator study: four operators, one of which logs 2 events. The
    # marginal MLE is interior (α̂ ≈ 1.65), but J22 mixes the observed ψ' term with expected
    # terms and comes out negative, so the asymptotic variance of α̂ does not exist.
    x = np.array([0, 0, 0, 2])
    t = np.array([0.84, 1.26, 1.68, 2.1])
    fit = fit_marginal_mle(x, t)
    assert not fit.pooled
    with pytest.raises(ValueError, match="information"):
        kass_steffey_posteriors(fit, x, t)


def test_analytic_score_matches_finite_differences() -> None:
    # The fit uses an analytic gradient; check it against central differences of the
    # marginal log-likelihood in (log mu, log alpha) at an arbitrary point.
    mu, alpha = 0.9, 1.7
    h = 1e-6

    def ll(lmu: float, la: float) -> float:
        a = np.exp(la)
        return marginal_loglik(a, a / np.exp(lmu), X, T)

    g_mu = (ll(np.log(mu) + h, np.log(alpha)) - ll(np.log(mu) - h, np.log(alpha))) / (2 * h)
    g_a = (ll(np.log(mu), np.log(alpha) + h) - ll(np.log(mu), np.log(alpha) - h)) / (2 * h)
    from scipy import special

    d = alpha + mu * T
    a_mu = np.sum(X - (alpha + X) * mu * T / d)
    a_a = alpha * np.sum(
        special.digamma(alpha + X)
        - special.digamma(alpha)
        + np.log(alpha)
        + 1
        - np.log(d)
        - (alpha + X) / d
    )
    assert a_mu == pytest.approx(g_mu, rel=1e-6)
    assert a_a == pytest.approx(g_a, rel=1e-6)
