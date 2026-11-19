import numpy as np
import pytest

from smallfleet.data_generation import (
    FleetHistory,
    gamma_population,
    load_components,
    load_fleets,
    monthly_exposure,
    simulate_counts,
    simulate_history,
)


def test_tables_load_with_a_basis_on_every_row() -> None:
    components = load_components()
    fleets = load_fleets()
    assert len(components) == 6
    assert {"business_jet", "commercial_narrowbody"} <= set(fleets)
    for spec in components.values():
        assert spec.large_fleet_rate > 0 and spec.small_fleet_rate > 0
        assert spec.basis.startswith("illustrative assumption")
    for fleet in fleets.values():
        assert fleet.basis.startswith("illustrative assumption")


def test_exposure_scales_with_fleet_size() -> None:
    exposure = monthly_exposure(5, 35.0, 12)
    assert exposure.shape == (12,)
    np.testing.assert_array_equal(exposure, 175.0)
    with pytest.raises(ValueError):
        monthly_exposure(0, 35.0, 12)


def test_counts_have_the_poisson_mean_and_variance() -> None:
    # Monte Carlo against the closed form: E[Y] = Var[Y] = λT/1000. With n = 200 000 the
    # standard error of the sample mean is sqrt(m/n); we allow 5 standard errors.
    rng = np.random.default_rng(1)
    exposure = np.full(200_000, 2000.0)
    y = simulate_counts(rng, 0.5, exposure)
    m = 1.0
    se = np.sqrt(m / y.size)
    assert abs(y.mean() - m) < 5 * se
    assert abs(y.var() - m) < 5 * np.sqrt(2 * m**2 / y.size + m / y.size)
    with pytest.raises(ValueError):
        simulate_counts(rng, -1.0, exposure)


def test_history_is_seed_deterministic_and_consistent() -> None:
    a = simulate_history(np.random.default_rng(7), 0.2, 5, 35.0, 24)
    b = simulate_history(np.random.default_rng(7), 0.2, 5, 35.0, 24)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.months == 24
    assert a.total_exposure_fh == pytest.approx(24 * 175.0)
    assert a.total_count == int(a.counts.sum())
    with pytest.raises(ValueError):
        FleetHistory(np.ones(3), np.ones(4, dtype=np.int64))


def test_gamma_population_has_the_requested_mean_and_cv() -> None:
    # Monte Carlo against the closed form: mean and cv of Gamma(1/cv², 1/(cv² mean)).
    rng = np.random.default_rng(3)
    rates = gamma_population(rng, 0.2, 0.5, 400_000)
    assert rates.mean() == pytest.approx(0.2, rel=5e-3)
    assert rates.std() / rates.mean() == pytest.approx(0.5, rel=1e-2)
    np.testing.assert_array_equal(gamma_population(rng, 0.2, 0.0, 3), 0.2)
    with pytest.raises(ValueError):
        gamma_population(rng, -0.2, 0.5, 3)
