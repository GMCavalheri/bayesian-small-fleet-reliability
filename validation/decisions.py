"""Extension: decisions under rate uncertainty (alerts from costs, waiting, intervals)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from smallfleet import plotting
from smallfleet.bayesian_alert_thresholds import alert_count_threshold, window_false_alarm_rate
from smallfleet.bayesian_point_estimation import GammaRate, jeffreys
from smallfleet.data_generation import PER, load_components, load_fleets
from smallfleet.decision_theory import (
    AlertCosts,
    bayes_threshold,
    interval_for_expected_unavailability,
    interval_for_known_rate,
    interval_for_risk,
    load_decision_parameters,
    value_of_waiting,
)

from common import FIGURES, table, write_result

COMPONENT = "starter_generator"
EQUIVALENT_FH = 10_000.0
WINDOW = 3
HIDDEN_PRIOR_FH = 50_000.0
YEARS = np.arange(1, 21)


def main() -> None:
    params = load_decision_parameters()
    comp = load_components()[COMPONENT]
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    rate = comp.small_fleet_rate
    target = 1.5 * rate
    prior = GammaRate(0.5 + comp.large_fleet_rate * EQUIVALENT_FH / PER, EQUIVALENT_FH / PER)
    costs = AlertCosts(
        params["cost_false_alarm"], params["cost_missed_degradation"], params["cost_delay"]
    )

    # 1. Alert thresholds from cost ratios (25 aircraft, 3-month window).
    month_fh = 25 * fh
    exposure = WINDOW * month_fh
    rows = []
    for ratio in (1.0, 4.0, 9.0, 19.0, 49.0):
        c_costs = AlertCosts(1.0, ratio)
        p_star = bayes_threshold(c_costs)
        c = alert_count_threshold(prior, exposure, target, p_star)
        fa = window_false_alarm_rate(prior, rate, month_fh, WINDOW, target, p_star)
        det = float(stats.poisson.sf(c - 1, 2 * rate * exposure / PER))
        rows.append([ratio, round(p_star, 4), c, float(f"{fa:.4g}"), float(f"{det:.4g}")])
    thresholds = table(
        [
            "c_miss / c_FA",
            "p* = c_FA/(c_FA + c_miss)",
            "alert count in 3 months",
            "false alarms per month",
            "alert probability per month at 2× rate (full window)",
        ],
        rows,
    )

    # 2. Investigate now, or wait a month? (5 and 25 aircraft, 3-month window history.)
    wait_rows = []
    for n in (5, 25):
        m_fh = n * fh
        for y in range(5):
            post = prior.update(y, WINDOW * m_fh)
            p = post.sf(target)
            v = value_of_waiting(post, target, m_fh, costs)
            act_now = "investigate" if p > bayes_threshold(costs) else "no action"
            choice = "defer one month" if v.wait_is_better else act_now
            wait_rows.append(
                [n, y, round(p, 3), act_now, round(v.evsi, 4), round(v.delay_cost, 4), choice]
            )
    waiting = table(
        [
            "aircraft",
            "events in last 3 months",
            "P(λ > target)",
            "best action now",
            "EVSI of one more month",
            "delay cost",
            "decision",
        ],
        wait_rows,
    )

    # 3. Failure-finding interval for a hidden failure (5 aircraft).
    h_rate = params["hidden_failure_rate_per_1000fh"]
    u_star = params["unavailability_target"]
    q = params["risk_quantile"]
    known = interval_for_known_rate(h_rate, u_star)
    h_prior = GammaRate(0.5 + h_rate * HIDDEN_PRIOR_FH / PER, HIDDEN_PRIOR_FH / PER)
    year_fh = 5 * fh * 12
    ffi_rows = []
    for years, y in ((3, 0), (3, 1), (3, 2), (10, 0), (10, 4)):
        t_fh = years * year_fh
        j = jeffreys().update(y, t_fh)
        p = h_prior.update(y, t_fh)
        mle = PER * y / t_fh
        ffi_rows.append(
            [
                years,
                y,
                "∞ (rate 0)" if y == 0 else round(interval_for_known_rate(mle, u_star)),
                round(interval_for_risk(j, u_star, q)),
                round(interval_for_expected_unavailability(p, u_star)),
                round(interval_for_risk(p, u_star, q)),
            ]
        )
    ffi = table(
        [
            "years of data (5 aircraft)",
            "hidden failures found",
            "plug-in MLE",
            f"Jeffreys, {q:.0%} risk",
            "large-fleet prior, expected U",
            f"large-fleet prior, {q:.0%} risk",
        ],
        ffi_rows,
    )

    # Expected risk-based interval against years of data, with and without the prior.
    curves: dict[str, list[float]] = {"Jeffreys": [], "large-fleet prior": []}
    for years in YEARS:
        t_fh = years * year_fh
        m = h_rate * t_fh / PER
        ys = np.arange(int(stats.poisson.ppf(1 - 1e-12, m)) + 2)
        w = stats.poisson.pmf(ys, m)
        for label, base in (("Jeffreys", jeffreys()), ("large-fleet prior", h_prior)):
            taus = np.array([interval_for_risk(base.update(int(y), t_fh), u_star, q) for y in ys])
            curves[label].append(float(np.sum(w * taus) / w.sum()))
    plotting.use_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        YEARS,
        curves["Jeffreys"],
        "o-",
        color=plotting.FREQUENTIST_COLOR,
        label="Jeffreys prior (own data only)",
    )
    ax.plot(
        YEARS,
        curves["large-fleet prior"],
        "o-",
        color=plotting.BAYES_COLOR,
        label=f"large-fleet prior ({HIDDEN_PRIOR_FH:,.0f} FH)",
    )
    ax.axhline(
        known, color=plotting.NEUTRAL_COLOR, ls="--", lw=1, label="interval if the rate were known"
    )
    ax.set_xlabel("years of data on 5 business jets")
    ax.set_ylabel(f"expected {q:.0%}-risk test interval (FH)")
    ax.set_ylim(0, known * 1.1)
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "decisions_interval.png", metadata={"Software": None})
    plt.close(fig)
    value_rows = [
        [int(yr), round(curves["Jeffreys"][i]), round(curves["large-fleet prior"][i])]
        for i, yr in enumerate(YEARS)
        if yr in (1, 2, 5, 10, 20)
    ]

    write_result(
        "decisions",
        "Extension: decisions under rate uncertainty",
        [
            (
                "Parameters",
                table(
                    ["parameter", "value"],
                    [[k, v] for k, v in params.items()],
                )
                + "\nEvery value is an illustrative assumption; the basis of each is in "
                "`src/smallfleet/data/decision_parameters.csv`.\n",
            ),
            (
                "Alert thresholds from costs",
                f"Component `{COMPONENT}` on 25 business jets, nominal rate {rate}, target "
                f"{target:g}, prior Gamma({prior.alpha:g}, {prior.beta:g}) as in Phase 4, "
                f"{WINDOW}-month window. All values exact.\n\n" + thresholds,
            ),
            (
                "Investigate now or wait a month",
                f"Costs: false alarm {costs.false_alarm:g}, missed degradation "
                f"{costs.missed:g}, delay {costs.delay:g}; threshold p* = "
                f"{bayes_threshold(costs):.2g}. The posterior uses the last {WINDOW} months. "
                "Waiting is chosen when the expected value of one more month of data exceeds "
                "the expected cost of the delay. All values exact.\n\n" + waiting,
            ),
            (
                "Failure-finding interval for a hidden failure",
                f"True rate {h_rate} per 1000 FH (interval {known:.0f} FH if it were known), "
                f"unavailability target {u_star:g}. The large-fleet prior counts the large "
                f"fleet's rate as {HIDDEN_PRIOR_FH:,.0f} FH: Gamma({h_prior.alpha:g}, "
                f"{h_prior.beta:g}). Intervals in FH.\n\n" + ffi,
            ),
            (
                "What data are worth",
                f"Expected {q:.0%}-risk interval (FH) over the Poisson outcomes at the true "
                "rate, by years of data on 5 aircraft (exact sums).\n\n"
                + table(["years", "Jeffreys", "large-fleet prior"], value_rows),
            ),
        ],
    )


if __name__ == "__main__":
    main()
