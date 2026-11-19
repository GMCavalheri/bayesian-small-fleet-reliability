"""Phase 3: hierarchical partial pooling, exact, against empirical Bayes and the extremes.

Writes ``results/hierarchical.md`` from the exact oracle only (byte-reproducible). The MCMC
comparison lives in ``hierarchical_mcmc.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting
from smallfleet.data_generation import gamma_population, load_components, load_fleets
from smallfleet.empirical_bayes import (
    fit_marginal_mle,
    kass_steffey_posteriors,
    plugin_posteriors,
)
from smallfleet.frequentist_baseline import garwood_interval
from smallfleet.hierarchical_model import MeanCVHyperprior, exact_posterior

from common import FIGURES, table, write_result

SEED = 20261121
COMPONENT = "starter_generator"
FLEET_SIZES = (2, 3, 4, 5, 6, 8, 10, 12)
MONTHS = 12
OPERATORS = (4, 8, 16, 32)
CVS = (0.25, 0.5)
REPLICATES = 300
WIDTH_REPLICATES = 40
MASS = 0.90
GRID = 121


def hyperprior() -> MeanCVHyperprior:
    """Population mean centred on the large fleet's rate (a factor of 5 either way at 90 %),
    spread weakly regularised towards zero."""
    return MeanCVHyperprior(
        mean_median=load_components()[COMPONENT].large_fleet_rate, mean_log_sd=1.0, cv_scale=1.0
    )


def operator_study() -> tuple[str, str, dict[tuple[float, str], list[float]]]:
    rng = np.random.default_rng(SEED)
    comp = load_components()[COMPONENT]
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    prior = hyperprior()
    rows, coverage_curves = [], {}
    for cv in CVS:
        for k in OPERATORS:
            t = np.resize(np.array(FLEET_SIZES), k) * fh * MONTHS / 1000
            stats: dict[str, dict[str, list[float]]] = {}

            def add(
                label: str,
                lam: np.ndarray,
                mean: np.ndarray,
                hit: np.ndarray,
                stats: dict[str, dict[str, list[float]]] = stats,
            ) -> None:
                s = stats.setdefault(label, {"sq": [], "hit": [], "width": []})
                s["sq"].extend(((mean - lam) / lam) ** 2)
                s["hit"].extend(hit)

            for rep in range(REPLICATES):
                lam = gamma_population(rng, comp.small_fleet_rate, cv, k)
                x = rng.poisson(lam * t)
                # no pooling
                g = np.array(
                    [
                        garwood_interval(int(xi), ti * 1000, MASS)
                        for xi, ti in zip(x, t, strict=True)
                    ]
                )
                add("no pooling", lam, x / t, (g[:, 0] <= lam) & (lam <= g[:, 1]))
                # hierarchical Bayes (exact)
                post = exact_posterior(x, t, prior, n=GRID)
                means = np.array([post.unit_mean(i) for i in range(k)])
                cdf = np.array([post.unit_cdf(i, float(lam[i])) for i in range(k)])
                a = (1 - MASS) / 2
                add("hierarchical Bayes", lam, means, (cdf > a) & (cdf < 1 - a))
                if rep < WIDTH_REPLICATES:
                    widths = [np.subtract(*post.unit_interval(i, MASS)[::-1]) for i in range(k)]
                    stats["hierarchical Bayes"]["width"].extend(np.array(widths) / lam)
                    stats["no pooling"]["width"].extend((g[:, 1] - g[:, 0]) / lam)
                # empirical Bayes (as in the empirical Bayes study)
                if x.sum() > 0:
                    fit = fit_marginal_mle(x, t)
                    plug = plugin_posteriors(fit, x, t)
                    try:
                        ks = kass_steffey_posteriors(fit, x, t)
                    except ValueError:
                        ks = plug
                    for label, posts in (
                        ("empirical Bayes, plug-in", plug),
                        ("empirical Bayes, Kass-Steffey", ks),
                    ):
                        iv = np.array([p.interval(MASS) for p in posts])
                        add(
                            label,
                            lam,
                            np.array([p.mean() for p in posts]),
                            (iv[:, 0] <= lam) & (lam <= iv[:, 1]),
                        )
                        if rep < WIDTH_REPLICATES:
                            stats[label]["width"].extend((iv[:, 1] - iv[:, 0]) / lam)
            for label, s in stats.items():
                cover = float(np.mean(s["hit"]))
                rows.append(
                    [
                        cv,
                        k,
                        label,
                        round(float(np.sqrt(np.mean(s["sq"]))), 3),
                        round(cover, 3),
                        round(float(np.median(s["width"])), 3) if s["width"] else "—",
                    ]
                )
                coverage_curves.setdefault((cv, label), []).append(cover)
    perf = table(
        ["population cv", "operators", "method", "RMSE / λ", "90 % coverage", "median width / λ"],
        rows,
    )
    setup = (
        f"Same setting as the empirical Bayes study (component `{COMPONENT}`, operators of "
        f"{', '.join(map(str, FLEET_SIZES))} business jets cycled, {MONTHS} months), with a new "
        f"seed ({SEED}) and {REPLICATES} replicates per cell; all methods see the same draws. "
        "Hierarchical Bayes is the exact posterior (oracle, "
        f"{GRID}×{GRID} grid) under the hyperprior μ ~ LogNormal(log 0.2, 1), cv ~ HalfNormal(1): "
        "the population mean is centred on the large fleet's rate, 0.2, which is 20 % below "
        "the true 0.25. Coverage of a unit is read from its exact posterior CDF at the true "
        "rate. Interval widths are medians over the first "
        f"{WIDTH_REPLICATES} replicates. Empirical Bayes is skipped in years with no events "
        "(its coverage is over the remaining years).\n"
    )
    return setup, perf, coverage_curves


def forest_example() -> str:
    """One illustrative draw: 8 operators, cv 0.5, all methods side by side."""
    rng = np.random.default_rng(SEED + 1)
    comp = load_components()[COMPONENT]
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    t = np.array(FLEET_SIZES) * fh * MONTHS / 1000
    lam = gamma_population(rng, comp.small_fleet_rate, 0.5, t.size)
    x = rng.poisson(lam * t)
    post = exact_posterior(x, t, hyperprior())
    fit = fit_marginal_mle(x, t)
    eb = plugin_posteriors(fit, x, t)
    rows = []
    plotting.use_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i in range(t.size):
        g = garwood_interval(int(x[i]), t[i] * 1000, MASS)
        h = post.unit_interval(i, MASS)
        e = eb[i].interval(MASS)
        rows.append(
            [
                f"operator {i + 1}",
                FLEET_SIZES[i],
                int(x[i]),
                round(float(lam[i]), 3),
                f"{x[i] / t[i]:.3f} ({g[0]:.3f}, {g[1]:.3f})",
                f"{eb[i].mean():.3f} ({e[0]:.3f}, {e[1]:.3f})",
                f"{post.unit_mean(i):.3f} ({h[0]:.3f}, {h[1]:.3f})",
            ]
        )
        y = t.size - i
        ax.plot(g, [y + 0.22] * 2, color=plotting.FREQUENTIST_COLOR, lw=2)
        ax.plot(x[i] / t[i], y + 0.22, "o", color=plotting.FREQUENTIST_COLOR)
        ax.plot(e, [y] * 2, color=plotting.BAYES_COLOR, lw=2)
        ax.plot(eb[i].mean(), y, "o", color=plotting.BAYES_COLOR)
        ax.plot(h, [y - 0.22] * 2, color=plotting.POOLED_COLOR, lw=2)
        ax.plot(post.unit_mean(i), y - 0.22, "o", color=plotting.POOLED_COLOR)
        ax.plot(lam[i], y, "k|", ms=16, mew=2)
    ax.set_yticks(
        range(1, t.size + 1),
        [f"op. {t.size - j} ({FLEET_SIZES[t.size - j - 1]} ac)" for j in range(t.size)],
    )
    ax.set_xlabel("starter-generator removals per 1000 FH (black tick: true rate)")
    ax.set_xlim(0, 2.5)
    ax.plot([], [], color=plotting.FREQUENTIST_COLOR, lw=2, label="no pooling (MLE + Garwood)")
    ax.plot([], [], color=plotting.BAYES_COLOR, lw=2, label="empirical Bayes, plug-in")
    ax.plot([], [], color=plotting.POOLED_COLOR, lw=2, label="hierarchical Bayes (exact)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGURES / "hierarchical_forest.png", metadata={"Software": None})
    plt.close(fig)
    m = post.hyper_moments()
    note = (
        f"One draw (seed {SEED + 1}): 8 operators, population cv 0.5, {int(x.sum())} events in "
        f"{t.sum() * 1000:,.0f} FH. Empirical Bayes fit: α = {fit.alpha:.3g}, "
        f"{'zero spread (pooled)' if fit.pooled else 'interior'}. Hierarchical posterior: "
        f"mean of μ {m['mu_mean']:.3f}, mean of cv {m['cv_mean']:.2f} (sd {m['cv_sd']:.2f}).\n"
    )
    return (
        note
        + "\n"
        + table(
            [
                "operator",
                "aircraft",
                "events",
                "true rate",
                "MLE (90 % Garwood)",
                "empirical Bayes (90 %)",
                "hierarchical Bayes (90 %)",
            ],
            rows,
        )
    )


def main() -> None:
    setup, perf, curves = operator_study()
    plotting.use_style()
    fig, axes = plt.subplots(1, len(CVS), figsize=(10, 3.8), sharey=True)
    labels = [
        "no pooling",
        "empirical Bayes, plug-in",
        "empirical Bayes, Kass-Steffey",
        "hierarchical Bayes",
    ]
    colors = (plotting.FREQUENTIST_COLOR, plotting.BAYES_COLOR, "#7a4fa3", plotting.POOLED_COLOR)
    for ax, cv in zip(axes, CVS, strict=True):
        for label, color in zip(labels, colors, strict=True):
            ax.plot(OPERATORS, curves[(cv, label)], "o-", color=color, label=label)
        ax.axhline(MASS, color=plotting.NEUTRAL_COLOR, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(OPERATORS, [str(k) for k in OPERATORS])
        ax.set_xlabel("operators pooled")
        ax.set_title(f"population cv = {cv}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("coverage of the 90 % interval")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.savefig(FIGURES / "hierarchical_coverage.png", metadata={"Software": None})
    plt.close(fig)
    write_result(
        "hierarchical",
        "Phase 3: hierarchical partial pooling (exact)",
        [
            ("One example", forest_example()),
            ("Operator study: setup", setup),
            ("Operator study: performance", perf),
        ],
    )


if __name__ == "__main__":
    main()
