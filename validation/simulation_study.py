"""Phase 5: when does borrowing strength help, and at what fleet size do methods converge?"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting
from smallfleet.data_generation import load_fleets
from smallfleet.simulation_study import (
    convergence_size,
    operator_pooling_point,
    single_fleet_point,
)

from common import FIGURES, table, write_result

SEED = 20261125
LARGE_RATE = 0.2
RATIOS = {"good prior (×1.0)": 1.0, "biased prior (×1.5)": 1.5, "conflicting prior (×3.0)": 3.0}
SIZES = (2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400)
POOL_SIZES = (2, 5, 12, 30, 80, 200)
MONTHS = 12
EQUIVALENT_FH = 10_000.0
LARGE_FH = 900_000.0
OPERATORS = 8
CV = 0.5
REPLICATES = 120


def main() -> None:
    fh = load_fleets()["business_jet"].fh_per_aircraft_month
    rng = np.random.default_rng(SEED)
    grid: dict[str, list] = {}
    pooled: dict[str, list] = {}
    rows, head_rows, pool_rows = [], [], []
    for label, ratio in RATIOS.items():
        truth = ratio * LARGE_RATE
        pts = [
            single_fleet_point(n, truth, LARGE_RATE, fh, MONTHS, EQUIVALENT_FH, LARGE_FH)
            for n in SIZES
        ]
        grid[label] = pts
        for p in pts:
            ref = p.methods["MLE + Garwood"]
            for m in ("MLE + Garwood", "Jeffreys", "borrowed prior", "full borrowing"):
                perf = p.methods[m]
                rows.append(
                    [
                        label,
                        p.aircraft,
                        m,
                        round(perf.rmse / p.true_rate, 3),
                        round(perf.rmse / ref.rmse, 3),
                        round(perf.coverage, 3),
                        round(perf.mean_width / p.true_rate, 3),
                    ]
                )
        pool = [
            operator_pooling_point(rng, n, OPERATORS, truth, CV, LARGE_RATE, fh, MONTHS, REPLICATES)
            for n in POOL_SIZES
        ]
        pooled[label] = pool
        for q in pool:
            r_mle, r_hb = q.rmse["MLE + Garwood"], q.rmse["hierarchical Bayes"]
            c_hb = q.coverage["hierarchical Bayes"]
            pool_rows.append(
                [
                    label,
                    q.aircraft,
                    f"{r_mle.value:.3f} ± {r_mle.se:.3f}",
                    f"{r_hb.value:.3f} ± {r_hb.se:.3f}",
                    f"{c_hb.value:.3f} ± {c_hb.se:.3f}",
                ]
            )
        ratios = [p.methods["borrowed prior"].rmse / p.methods["MLE + Garwood"].rmse for p in pts]
        best = int(np.argmin(ratios))
        five = next(p for p in pts if p.aircraft == 5)
        conv = convergence_size(pts, "borrowed prior")
        head_rows.append(
            [
                label,
                round(ratios[SIZES.index(5)], 3),
                f"{ratios[best]:.3f} at {SIZES[best]}",
                round(five.methods["borrowed prior"].coverage, 3),
                round(five.methods["full borrowing"].coverage, 3),
                "beyond 400" if conv is None else conv,
            ]
        )

    plotting.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (label, pts), color in zip(grid.items(), plotting.SERIES_COLORS[1:4], strict=True):
        ref = np.array([p.methods["MLE + Garwood"].rmse for p in pts])
        bp = np.array([p.methods["borrowed prior"].rmse for p in pts])
        fb = np.array([p.methods["full borrowing"].rmse for p in pts])
        ax1.plot(SIZES, bp / ref, "-", color=color, label=f"borrowed prior, {label}")
        ax1.plot(SIZES, fb / ref, ":", color=color)
        ax2.plot(SIZES, [p.methods["borrowed prior"].coverage for p in pts], "-", color=color)
        ax2.plot(SIZES, [p.methods["full borrowing"].coverage for p in pts], ":", color=color)
        pool = pooled[label]
        hb = [q.rmse["hierarchical Bayes"].value / q.rmse["MLE + Garwood"].value for q in pool]
        ax1.plot(POOL_SIZES, hb, "s", color=color, ms=5)
        ax2.errorbar(
            POOL_SIZES,
            [q.coverage["hierarchical Bayes"].value for q in pool],
            yerr=[2 * q.coverage["hierarchical Bayes"].se for q in pool],
            fmt="s",
            color=color,
            ms=5,
            capsize=2,
        )
    ax1.axhline(1.0, color=plotting.NEUTRAL_COLOR, lw=1)
    ax1.axhspan(0.9, 1.1, color=plotting.NEUTRAL_COLOR, alpha=0.08)
    ax2.axhline(0.90, color=plotting.NEUTRAL_COLOR, lw=1)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("fleet size (aircraft, one year)")
    ax1.set_ylabel("RMSE relative to the MLE")
    ax1.set_ylim(0, 2.5)
    ax1.set_title("solid: borrowed prior · dotted: full borrowing · squares: hierarchical")
    ax1.title.set_fontsize(9)
    ax2.set_ylabel("coverage of the 90 % interval")
    ax2.set_ylim(0, 1.05)
    ax1.legend(fontsize=7, loc="upper left")
    fig.savefig(FIGURES / "simulation_study.png", metadata={"Software": None})
    plt.close(fig)

    write_result(
        "simulation_study",
        "Phase 5: when does Bayesian borrowing help?",
        [
            (
                "Setup",
                f"One component on business jets ({fh:g} FH per aircraft-month) observed for "
                f"{MONTHS} months; the large fleet's rate is {LARGE_RATE} per 1000 FH and the "
                "small fleet's true rate is 1.0, 1.5 or 3.0 times it (good, biased and "
                f"conflicting prior). The borrowed prior has the large fleet's mean and is worth "
                f"{EQUIVALENT_FH:,.0f} FH; full borrowing counts all {LARGE_FH:,.0f} FH. "
                "Single-fleet results are exact (sums over the Poisson count). The hierarchical "
                f"arm pools {OPERATORS} operators of the same size whose rates vary with cv {CV} "
                f"around the true mean, under μ ~ LogNormal(log {LARGE_RATE}, 1), cv ~ "
                f"HalfNormal(1); it is seeded Monte Carlo ({REPLICATES} replicates, seed {SEED}), "
                "reported with standard errors.\n",
            ),
            (
                "Headline",
                "RMSE ratio is the borrowed prior's RMSE divided by the MLE's. Convergence is the "
                "smallest fleet size from which the ratio stays within 10 % of 1.\n\n"
                + table(
                    [
                        "prior quality",
                        "RMSE ratio at 5 aircraft",
                        "best RMSE ratio",
                        "coverage at 5, borrowed",
                        "coverage at 5, full borrowing",
                        "converges at (aircraft)",
                    ],
                    head_rows,
                ),
            ),
            (
                "Hierarchical pooling across operators",
                "Per-operator relative RMSE and coverage, mean ± one standard error.\n\n"
                + table(
                    [
                        "prior quality",
                        "aircraft per operator",
                        "RMSE / λ, MLE",
                        "RMSE / λ, hierarchical",
                        "90 % coverage, hierarchical",
                    ],
                    pool_rows,
                ),
            ),
            (
                "Single-fleet grid",
                table(
                    [
                        "prior quality",
                        "aircraft",
                        "method",
                        "RMSE / λ",
                        "RMSE / MLE RMSE",
                        "90 % coverage",
                        "mean width / λ",
                    ],
                    rows,
                ),
            ),
        ],
    )


if __name__ == "__main__":
    main()
