"""Phase 2: what a prior borrowed from a large fleet buys a 5-aircraft fleet, and what it costs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from smallfleet import plotting
from smallfleet.bayesian_point_estimation import (
    bayes_estimator,
    exact_performance,
    frequentist_estimator,
    jeffreys,
    power_prior,
    prior_predictive_pvalue,
)
from smallfleet.data_generation import load_components, load_fleets, simulate_history

from common import FIGURES, table, write_result

SEED = 20261119
SMALL_AIRCRAFT, SMALL_MONTHS = 5, 12
LARGE_AIRCRAFT, LARGE_MONTHS = 100, 36
EQUIVALENT_FH = 10_000.0
SWEEP = np.geomspace(500.0, 900_000.0, 25)


def conflict_power(prior, rate: float, exposure_fh: float, level: float = 0.05) -> float:
    """Exact probability that the prior-data conflict p-value falls below ``level``."""
    m = rate * exposure_fh / 1000
    ys = np.arange(int(stats.poisson.ppf(1 - 1e-14, m)) + 2)
    w = stats.poisson.pmf(ys, m)
    flags = np.array([prior_predictive_pvalue(prior, int(y), exposure_fh) < level for y in ys])
    return float(w @ flags / w.sum())


def main() -> None:
    rng = np.random.default_rng(SEED)
    comps = load_components()
    fleets = load_fleets()
    small_fh = SMALL_AIRCRAFT * fleets["business_jet"].fh_per_aircraft_month * SMALL_MONTHS
    large = {
        name: simulate_history(
            rng,
            c.large_fleet_rate,
            LARGE_AIRCRAFT,
            fleets["commercial_narrowbody"].fh_per_aircraft_month,
            LARGE_MONTHS,
        )
        for name, c in comps.items()
    }
    large_fh = next(iter(large.values())).total_exposure_fh

    prior_rows, perf_rows, conflict_rows = [], [], []
    for name, c in comps.items():
        h = large[name]
        weak = power_prior(h.total_count, h.total_exposure_fh, EQUIVALENT_FH)
        full = power_prior(h.total_count, h.total_exposure_fh, h.total_exposure_fh)
        prior_rows.append(
            [
                name,
                c.large_fleet_rate,
                c.small_fleet_rate,
                h.total_count,
                round(weak.mean(), 4),
                f"({weak.interval()[0]:.3f}, {weak.interval()[1]:.3f})",
                f"({full.interval()[0]:.3f}, {full.interval()[1]:.3f})",
            ]
        )
        methods = {
            "MLE + Garwood": frequentist_estimator(),
            "Jeffreys": bayes_estimator(jeffreys()),
            f"power prior, {EQUIVALENT_FH:,.0f} FH": bayes_estimator(weak),
            "full borrowing": bayes_estimator(full),
        }
        for label, est in methods.items():
            p = exact_performance(est, c.small_fleet_rate, small_fh)
            perf_rows.append(
                [
                    name,
                    label,
                    round(p.rmse / c.small_fleet_rate, 3),
                    round(p.bias / c.small_fleet_rate, 3) + 0.0,
                    round(p.coverage, 3),
                    round(p.mean_width / c.small_fleet_rate, 3),
                ]
            )
        conflict_rows.append(
            [
                name,
                round(c.small_fleet_rate / c.large_fleet_rate, 2),
                round(conflict_power(weak, c.small_fleet_rate, small_fh), 3),
                round(conflict_power(full, c.small_fleet_rate, small_fh), 3),
                round(conflict_power(full, c.small_fleet_rate, 5 * small_fh), 3),
            ]
        )

    priors = table(
        [
            "component",
            "large-fleet rate",
            "small-fleet rate",
            "large-fleet events",
            "power-prior mean",
            "power-prior 90 %",
            "full-borrowing 90 %",
        ],
        prior_rows,
    )
    perf = table(
        ["component", "method", "RMSE / λ", "bias / λ", "90 % coverage", "mean width / λ"],
        perf_rows,
    )
    conflict = table(
        [
            "component",
            "small / large rate",
            "P(flag), power prior, 1 year",
            "P(flag), full borrowing, 1 year",
            "P(flag), full borrowing, 5 years",
        ],
        conflict_rows,
    )

    plotting.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, color in zip(
        ("fuel_boost_pump", "starter_generator", "bleed_air_valve"),
        plotting.SERIES_COLORS[1:4],
        strict=True,
    ):
        c, h = comps[name], large[name]
        rmse, cov = [], []
        for eq in SWEEP:
            pr = power_prior(h.total_count, h.total_exposure_fh, float(eq))
            p = exact_performance(bayes_estimator(pr), c.small_fleet_rate, small_fh)
            rmse.append(p.rmse / c.small_fleet_rate)
            cov.append(p.coverage)
        ratio = c.small_fleet_rate / c.large_fleet_rate
        label = f"{name} (small/large = {ratio:.2f})"
        ax1.plot(SWEEP, rmse, color=color, label=label)
        ax2.plot(SWEEP, cov, color=color, label=label)
        mle = exact_performance(frequentist_estimator(), c.small_fleet_rate, small_fh)
        ax1.axhline(mle.rmse / c.small_fleet_rate, color=color, ls=":", lw=1)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("large-fleet exposure counted (equivalent FH)")
        ax.axvline(small_fh, color=plotting.NEUTRAL_COLOR, ls="--", lw=1)
    ax1.set_ylabel("RMSE / true rate (dotted: MLE)")
    ax1.set_title(f"{SMALL_AIRCRAFT} aircraft, {SMALL_MONTHS} months")
    ax2.set_ylabel("coverage of the 90 % interval")
    ax2.axhline(0.90, color=plotting.NEUTRAL_COLOR, lw=1)
    ax2.legend(fontsize=7)
    fig.savefig(FIGURES / "bayes_point.png", metadata={"Software": None})
    plt.close(fig)

    write_result(
        "bayes_point",
        "Phase 2: Bayesian estimation with a prior borrowed from a large fleet",
        [
            (
                "Setup",
                f"The small fleet is {SMALL_AIRCRAFT} business jets observed for "
                f"{SMALL_MONTHS} months ({small_fh:,.0f} FH). The large fleet is "
                f"{LARGE_AIRCRAFT} narrow-bodies over {LARGE_MONTHS} months "
                f"({large_fh:,.0f} FH), simulated once at the large-fleet rates (seed {SEED}). "
                "The power prior is the Jeffreys prior updated with the large fleet's data "
                f"counted as {EQUIVALENT_FH:,.0f} FH (a0 = {EQUIVALENT_FH / large_fh:.4f}); "
                "full borrowing counts all of it (a0 = 1). Every performance number is exact: "
                "a sum over the Poisson count at the true small-fleet rate.\n",
            ),
            ("Priors", priors),
            ("Repeated-sampling performance", perf),
            (
                "Prior-data conflict check",
                "Probability that the two-sided prior-predictive p-value falls below 0.05, "
                "at the true small-fleet rate (exact).\n\n" + conflict,
            ),
        ],
    )


if __name__ == "__main__":
    main()
