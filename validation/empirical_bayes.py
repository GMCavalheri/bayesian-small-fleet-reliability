"""Empirical Bayes across operators: an OEM pools several small fleets of one type."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting
from smallfleet.bayesian_point_estimation import GammaRate, jeffreys
from smallfleet.data_generation import gamma_population, load_components, load_fleets
from smallfleet.empirical_bayes import (
    EmpiricalBayesFit,
    fit_marginal_mle,
    kass_steffey_posteriors,
    method_of_moments,
    plugin_posteriors,
)
from smallfleet.frequentist_baseline import garwood_interval

from common import FIGURES, table, write_result

SEED = 20261120
COMPONENT = "starter_generator"
FLEET_SIZES = (2, 3, 4, 5, 6, 8, 10, 12)
"""Aircraft per operator, cycled to build K operators (illustrative)."""
MONTHS = 12
OPERATORS = (4, 8, 16, 32)
CVS = (0.25, 0.5)
REPLICATES = 500
MASS = 0.90
EB_LABELS = (
    "empirical Bayes, plug-in",
    "empirical Bayes, Kass-Steffey",
    "empirical Bayes, moments",
)


def methods(
    x: np.ndarray, t: np.ndarray, fit: EmpiricalBayesFit | None
) -> dict[str, list[tuple[float, float, float]]]:
    """Per-operator (point, lower, upper) for every method, rates per 1000 FH.

    When no operator logs an event, empirical Bayes has no estimate; every empirical Bayes
    method then falls back to the Jeffreys posterior of the pooled data, Gamma(1/2, Σt).
    """
    tot_x, tot_t = int(x.sum()), float(t.sum())
    pooled_lo, pooled_hi = garwood_interval(tot_x, tot_t * 1000, MASS)
    out: dict[str, list[tuple[float, float, float]]] = {
        "no pooling (MLE + Garwood)": [
            (xi / ti, *garwood_interval(int(xi), ti * 1000, MASS))
            for xi, ti in zip(x, t, strict=True)
        ],
        "complete pooling": [(tot_x / tot_t, pooled_lo, pooled_hi)] * len(x),
    }
    if fit is None:
        fallback = jeffreys().update(0, tot_t * 1000)
        est = [(fallback.mean(), *fallback.interval(MASS))] * len(x)
        for label in EB_LABELS:
            out[label] = est
        return out
    plug = plugin_posteriors(fit, x, t)
    try:
        ks = kass_steffey_posteriors(fit, x, t)
    except ValueError:  # zero spread or non-positive information: nothing to adjust
        ks = plug
    mom = method_of_moments(x, t)
    for label, posts in zip(EB_LABELS, (plug, ks, plugin_posteriors(mom, x, t)), strict=True):
        out[label] = [(p.mean(), *p.interval(MASS)) for p in posts]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    comp = load_components()[COMPONENT]
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    mean = comp.small_fleet_rate
    rows = []
    curves: dict[tuple[float, str], list[float]] = {}
    coverage: dict[tuple[float, str], list[float]] = {}
    pooled_share: dict[tuple[int, float], float] = {}
    empty_share: dict[tuple[int, float], float] = {}
    for cv in CVS:
        for k in OPERATORS:
            sizes = np.resize(np.array(FLEET_SIZES), k)
            t = sizes * fh * MONTHS / 1000
            sq: dict[str, list[float]] = {}
            hit: dict[str, list[float]] = {}
            width: dict[str, list[float]] = {}
            n_pooled = n_empty = 0
            for _ in range(REPLICATES):
                lam = gamma_population(rng, mean, cv, k)
                x = rng.poisson(lam * t)
                fit = None
                if x.sum() == 0:
                    n_empty += 1
                else:
                    fit = fit_marginal_mle(x, t)
                    n_pooled += int(fit.pooled)
                for label, est in methods(x, t, fit).items():
                    e = np.array(est)
                    sq.setdefault(label, []).extend(((e[:, 0] - lam) / lam) ** 2)
                    hit.setdefault(label, []).extend((e[:, 1] <= lam) & (lam <= e[:, 2]))
                    width.setdefault(label, []).extend((e[:, 2] - e[:, 1]) / lam)
            pooled_share[(k, cv)] = n_pooled / REPLICATES
            empty_share[(k, cv)] = n_empty / REPLICATES
            for label in sq:
                rrmse = float(np.sqrt(np.mean(sq[label])))
                rows.append(
                    [
                        cv,
                        k,
                        label,
                        round(rrmse, 3),
                        round(float(np.mean(hit[label])), 3),
                        round(float(np.median(width[label])), 3),
                    ]
                )
                curves.setdefault((cv, label), []).append(rrmse)
                coverage.setdefault((cv, label), []).append(float(np.mean(hit[label])))

    perf = table(
        ["population cv", "operators", "method", "RMSE / λ", "90 % coverage", "median width / λ"],
        rows,
    )
    pooled = table(
        [
            "operators",
            *[f"no events, cv = {cv}" for cv in CVS],
            *[f"zero spread, cv = {cv}" for cv in CVS],
        ],
        [
            [
                k,
                *[round(empty_share[(k, cv)], 3) for cv in CVS],
                *[round(pooled_share[(k, cv)], 3) for cv in CVS],
            ]
            for k in OPERATORS
        ],
    )

    plotting.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = sorted({label for _, label in curves})
    cv = CVS[-1]
    for label, color in zip(labels, plotting.SERIES_COLORS, strict=True):
        ax1.plot(OPERATORS, curves[(cv, label)], "o-", color=color, label=label)
        ax2.plot(OPERATORS, coverage[(cv, label)], "o-", color=color, label=label)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(OPERATORS, [str(k) for k in OPERATORS])
        ax.set_xlabel("operators pooled")
    ax1.set_ylabel("RMSE / true operator rate")
    ax1.set_ylim(0, 2.0)
    ax1.set_title(f"accuracy, population cv = {cv}")
    ax2.axhline(MASS, color=plotting.NEUTRAL_COLOR, lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("coverage of the 90 % interval")
    ax2.set_title("honesty of the interval")
    ax2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(FIGURES / "empirical_bayes.png", metadata={"Software": None})
    plt.close(fig)

    prior = GammaRate(1 / CVS[1] ** 2, 1 / CVS[1] ** 2 / mean)
    write_result(
        "empirical_bayes",
        "Empirical Bayes: pooling several small operators of one type",
        [
            (
                "Setup",
                f"Component `{COMPONENT}`. Each operator's true rate is drawn from a Gamma "
                f"population with mean {mean} per 1000 FH and coefficient of variation cv "
                "(illustrative assumption: operators differ by sector length, climate and "
                "maintenance practice). Operators fly "
                f"{', '.join(map(str, FLEET_SIZES))} aircraft (cycled) for {MONTHS} months at "
                f"{fh:g} FH per aircraft-month. Seeded Monte Carlo, {REPLICATES} replicates "
                "per cell; RMSE and coverage are pooled over operators and replicates "
                "(standard error of a coverage near 0.9 is at most 0.006). With cv = 0.5 the "
                f"population 90 % range is ({prior.ppf(0.05):.3f}, {prior.ppf(0.95):.3f}).\n",
            ),
            ("Performance", perf),
            (
                "Degenerate fits",
                "Share of replicates where no operator logs an event (empirical Bayes has no "
                "estimate; it falls back to the Jeffreys posterior of the pooled data) and "
                "where the marginal likelihood is maximised at zero spread (empirical Bayes "
                "collapses to complete pooling).\n\n" + pooled,
            ),
        ],
    )


if __name__ == "__main__":
    main()
