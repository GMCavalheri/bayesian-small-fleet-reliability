"""Phase 1: how the classical alert level and the classical interval degrade with fleet size."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting
from smallfleet.data_generation import load_components, load_fleets
from smallfleet.frequentist_baseline import (
    degenerate_baseline_probability,
    expected_relative_width,
    garwood_coverage,
    mean_sd_operating_characteristics,
    zero_month_probability,
)

from common import FIGURES, table, write_result

FLEET_SIZES = (3, 5, 10, 15, 25, 50, 100, 200)
COMPONENT = "starter_generator"
SEED = 20261118


def main() -> None:
    comp = load_components()[COMPONENT]
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    rate = comp.small_fleet_rate
    rng = np.random.default_rng(SEED)

    params = table(
        ["component", "ATA", "large-fleet rate", "small-fleet rate", "basis"],
        [
            [c.name, c.ata_chapter, c.large_fleet_rate, c.small_fleet_rate, c.basis]
            for c in load_components().values()
        ],
    )
    fleets = table(
        ["fleet", "FH per aircraft-month", "basis"],
        [[f.name, f.fh_per_aircraft_month, f.basis] for f in load_fleets().values()],
    )

    rows, widths, fa2, fa3, det2, degen = [], [], [], [], [], []
    for n in FLEET_SIZES:
        month_fh = n * fh
        m = rate * month_fh / 1000
        oc2 = mean_sd_operating_characteristics(rng, rate, month_fh, k=2.0)
        oc3 = mean_sd_operating_characteristics(rng, rate, month_fh, k=3.0)
        w = expected_relative_width(rate, 12 * month_fh, 0.90)
        pd = degenerate_baseline_probability(rate, month_fh)
        rows.append(
            [
                n,
                round(m, 4),
                round(zero_month_probability(rate, month_fh), 3),
                round(pd, 3),
                round(w, 2),
                round(garwood_coverage(rate, 12 * month_fh, 0.90), 3),
                round(oc2.false_alarm_rate, 3),
                round(oc3.false_alarm_rate, 3),
                round(oc2.detection_probability, 2),
            ]
        )
        widths.append(w)
        fa2.append(oc2.false_alarm_rate)
        fa3.append(oc3.false_alarm_rate)
        det2.append(oc2.detection_probability)
        degen.append(pd)

    study = table(
        [
            "aircraft",
            "events per month",
            "P(zero month)",
            "P(sd = 0 baseline)",
            "12-month 90 % interval width / λ",
            "exact coverage",
            "false alarms per month, 2σ",
            "false alarms per month, 3σ",
            "P(detect 2× within 3 months), 2σ",
        ],
        rows,
    )
    note = (
        f"Component `{COMPONENT}` at its true small-fleet rate of {rate} per 1000 FH, "
        f"{fh:g} FH per aircraft-month. Probabilities of a zero month, of a degenerate "
        "baseline, the interval width and the coverage are exact (closed forms and Poisson "
        "sums). False-alarm and detection rates are seeded Monte Carlo with 20 000 replicates "
        "(standard error at most 0.002 for false alarms and 0.004 for detection). A false "
        "alarm is a monitored month above a level set on the 12 previous months at the same "
        "true rate; a detection is any alert in the 3 months after the rate doubles.\n"
    )

    big = load_fleets()["commercial_narrowbody"].fh_per_aircraft_month
    big_rate = comp.large_fleet_rate
    big_fh = 100 * big
    oc_big = mean_sd_operating_characteristics(rng, big_rate, big_fh, k=2.0)
    reference = table(
        ["fleet", "events per month", "12-month 90 % interval width / λ", "false alarms, 2σ"],
        [
            [
                "100 narrow-bodies",
                round(big_rate * big_fh / 1000, 3),
                round(expected_relative_width(big_rate, 12 * big_fh, 0.90), 3),
                round(oc_big.false_alarm_rate, 3),
            ]
        ],
    )

    plotting.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(FLEET_SIZES, widths, "o-", color=plotting.FREQUENTIST_COLOR)
    ax1.set_xscale("log")
    ax1.set_xlabel("fleet size (aircraft)")
    ax1.set_ylabel("expected 90 % interval width / true rate")
    ax1.set_title("12-month Garwood interval")
    ax2.plot(FLEET_SIZES, fa2, "o-", color=plotting.FREQUENTIST_COLOR, label="false alarms, 2σ")
    ax2.plot(FLEET_SIZES, fa3, "s--", color=plotting.FREQUENTIST_COLOR, label="false alarms, 3σ")
    ax2.plot(FLEET_SIZES, degen, "^:", color=plotting.NEUTRAL_COLOR, label="P(sd = 0 baseline)")
    ax2.plot(FLEET_SIZES, det2, "d-", color=plotting.BAYES_COLOR, label="P(detect 2×), 2σ")
    ax2.set_xscale("log")
    ax2.set_xlabel("fleet size (aircraft)")
    ax2.set_ylabel("probability")
    ax2.set_title("mean + kσ alert level")
    ax2.legend(fontsize=8)
    fig.savefig(FIGURES / "frequentist_baseline.png", metadata={"Software": None})
    plt.close(fig)

    write_result(
        "frequentist",
        "Phase 1: the classical method on small fleets",
        [
            ("Components", params),
            ("Fleets", fleets),
            ("Instability against fleet size", note + "\n" + study),
            (
                "Large-fleet reference",
                "The same component on 100 narrow-bodies at the large-fleet rate, the setting "
                "the classical method was built for.\n\n" + reference,
            ),
        ],
    )


if __name__ == "__main__":
    main()
