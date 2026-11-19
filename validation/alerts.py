"""Phase 4: the Bayesian alert against the classical alert level, on the same histories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting
from smallfleet.bayesian_alert_thresholds import (
    alert_count_threshold,
    bayes_window_rule,
    characteristics,
    classical_rule,
    simulate_histories,
    window_false_alarm_rate,
)
from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.data_generation import load_components, load_fleets

from common import FIGURES, table, write_result

SEED = 20261124
COMPONENT = "starter_generator"
FLEET_SIZES = (3, 5, 10, 25, 50, 100)
BASELINE, MONITOR = 12, 12
SHIFT = 2.0
TARGET_FACTOR = 1.5
EQUIVALENT_FH = 10_000.0
REPLICATES = 4000
ROC_FLEETS = (5, 25)


def main() -> None:
    comp = load_components()[COMPONENT]
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    rate = comp.small_fleet_rate
    target = TARGET_FACTOR * rate
    # Power prior from the large fleet's expected data, discounted to EQUIVALENT_FH:
    # Gamma(1/2 + large rate × 10, 10).
    prior = GammaRate(0.5 + comp.large_fleet_rate * EQUIVALENT_FH / 1000, EQUIVALENT_FH / 1000)
    rng = np.random.default_rng(SEED)
    rules = {
        "classical, mean + 2σ": classical_rule(2.0),
        "classical, mean + 3σ": classical_rule(3.0),
        "Bayes, 3-month window, p* = 0.5": bayes_window_rule(prior, target, 0.5, 3),
        "Bayes, 6-month window, p* = 0.5": bayes_window_rule(prior, target, 0.5, 6),
        "Bayes, 12-month window, p* = 0.5": bayes_window_rule(prior, target, 0.5, 12),
    }
    rows, exact_rows = [], []
    roc: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for n in FLEET_SIZES:
        month_fh = n * fh
        hist = simulate_histories(rng, rate, month_fh, BASELINE, MONITOR, SHIFT, REPLICATES)
        for label, rule in rules.items():
            c = characteristics(hist, rule)
            rows.append(
                [
                    n,
                    label,
                    round(c.false_alarm_rate, 3),
                    round(c.detection, 3),
                    "—" if np.isnan(c.median_delay) else c.median_delay,
                ]
            )
        for w, p_star in ((3, 0.5), (6, 0.5), (12, 0.5)):
            exact_rows.append(
                [
                    n,
                    w,
                    p_star,
                    alert_count_threshold(prior, w * month_fh, target, p_star),
                    float(
                        f"{window_false_alarm_rate(prior, rate, month_fh, w, target, p_star):.4g}"
                    ),
                ]
            )
        if n in ROC_FLEETS:
            curves: dict[str, list[tuple[float, float]]] = {}
            for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                c = characteristics(hist, classical_rule(k))
                curves.setdefault("classical, mean + kσ", []).append(
                    (c.false_alarm_rate, c.detection)
                )
            for w in (3, 12):
                for p_star in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
                    c = characteristics(hist, bayes_window_rule(prior, target, p_star, w))
                    curves.setdefault(f"Bayes, {w}-month window", []).append(
                        (c.false_alarm_rate, c.detection)
                    )
            roc[n] = curves

    plotting.use_style()
    fig, axes = plt.subplots(1, len(ROC_FLEETS), figsize=(10, 3.8), sharey=True)
    colors = (plotting.FREQUENTIST_COLOR, plotting.BAYES_COLOR, plotting.POOLED_COLOR)
    for ax, n in zip(axes, ROC_FLEETS, strict=True):
        for (label, pts), color in zip(roc[n].items(), colors, strict=True):
            pts_sorted = sorted(p for p in set(pts) if p[0] > 0)
            ax.plot(
                [p[0] for p in pts_sorted],
                [p[1] for p in pts_sorted],
                "o-",
                color=color,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per month (rate unchanged)")
        ax.set_title(f"{n} aircraft")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(f"P(alert within {MONITOR} months of a {SHIFT:g}× rise)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.savefig(FIGURES / "alerts_roc.png", metadata={"Software": None})
    plt.close(fig)

    roc_rows = [
        [n, label, round(fa, 4), round(d, 3)]
        for n, curves in roc.items()
        for label, pts in curves.items()
        for fa, d in sorted(set(pts))
    ]
    write_result(
        "alerts",
        "Phase 4: Bayesian alert thresholds against the classical alert level",
        [
            (
                "Setup",
                f"Component `{COMPONENT}`, true rate {rate} per 1000 FH, {fh:g} FH per "
                f"aircraft-month. Each history has {BASELINE} baseline months and {MONITOR} "
                f"monitored months; in the shifted histories the rate is {SHIFT:g}× from the "
                f"first monitored month. The Bayesian rule alerts when P(λ > {target:g} | prior, "
                f"last w months) > p*; the target is {TARGET_FACTOR:g}× the nominal rate (an "
                "illustrative acceptance limit). Its prior is the large fleet's rate "
                f"({comp.large_fleet_rate}) counted as {EQUIVALENT_FH:,.0f} FH: "
                f"Gamma({prior.alpha:g}, {prior.beta:g}). The classical rule uses mean + kσ "
                f"of the {BASELINE} baseline monthly rates. Seeded Monte Carlo, {REPLICATES} "
                "replicates per fleet size, same histories for every rule (standard error of a "
                "false-alarm rate p is sqrt(p(1−p)/48 000), of a detection probability "
                "sqrt(p(1−p)/4000)).\n",
            ),
            (
                "Operating characteristics",
                table(
                    [
                        "aircraft",
                        "rule",
                        "false alarms per month",
                        f"P(detect within {MONITOR} months)",
                        "median delay (months)",
                    ],
                    rows,
                ),
            ),
            (
                "Exact false-alarm rate of the Bayesian rule",
                "With a full window the rule alerts when the window's count reaches c; at the "
                "nominal rate the count is Poisson, so the false-alarm rate is exact.\n\n"
                + table(
                    [
                        "aircraft",
                        "window (months)",
                        "p*",
                        "alert count c",
                        "false alarms per month",
                    ],
                    exact_rows,
                ),
            ),
            (
                "Detection at matched false-alarm rates",
                "Each rule's threshold swept (k for the classical rule, p* for the Bayesian "
                "one) on the same histories.\n\n"
                + table(["aircraft", "rule", "false alarms per month", "P(detect)"], roc_rows),
            ),
        ],
    )


if __name__ == "__main__":
    main()
