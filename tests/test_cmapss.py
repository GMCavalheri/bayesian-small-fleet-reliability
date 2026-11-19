from pathlib import Path

import numpy as np
import pytest

from smallfleet.cmapss import load_lifetimes, stationary_counts, true_rate

FIXTURE = Path(__file__).resolve().parent / "data" / "cmapss"


def test_loader_reads_lives_from_a_cmapss_format_fixture() -> None:
    # Synthetic fixture in the C-MAPSS layout (26 space-separated columns, trailing space):
    # three units with lives 5, 3 and 4 cycles.
    lives = load_lifetimes(FIXTURE, "FD001")
    np.testing.assert_array_equal(lives, [5, 3, 4])
    with pytest.raises(ValueError):
        load_lifetimes(FIXTURE, "FD009")


def test_true_rate_is_the_renewal_theorem_rate() -> None:
    assert true_rate(np.array([200, 300])) == pytest.approx(4.0)


def test_stationary_counts_have_mean_window_over_mean_life() -> None:
    # Stationary renewal process: E[N(T)] = positions · T / μ exactly, for any T (not only
    # asymptotically). Monte Carlo, 5 standard errors.
    lives = np.array([150, 200, 250, 300, 400])
    rng = np.random.default_rng(1)
    for window in (50, 500):
        counts = np.array([stationary_counts(rng, lives, 3, window) for _ in range(20_000)])
        expected = 3 * window / lives.mean()
        assert abs(counts.mean() - expected) < 5 * counts.std() / np.sqrt(counts.size)


def test_wear_out_lives_give_under_dispersed_counts() -> None:
    # Nearly constant lives make counts far less variable than Poisson with the same mean,
    # the misspecification the C-MAPSS study probes.
    lives = np.array([195, 200, 205])
    rng = np.random.default_rng(2)
    counts = np.array([stationary_counts(rng, lives, 2, 1000) for _ in range(5000)])
    assert counts.var() < 0.2 * counts.mean()
    with pytest.raises(ValueError):
        stationary_counts(rng, lives, 0, 100)
