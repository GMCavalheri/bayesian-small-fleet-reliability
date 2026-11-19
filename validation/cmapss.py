"""Extension: C-MAPSS lifetimes as four fleets with a known long-run rate.

Needs ``scripts/fetch_cmapss.py`` first (data are not in the repository); without the data
the script says so and exits, which is how the reproduce job skips it.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting
from smallfleet.cmapss import SUBSETS, load_lifetimes, stationary_counts, true_rate
from smallfleet.frequentist_baseline import garwood_interval
from smallfleet.hierarchical_model import MeanCVHyperprior, exact_posterior

from common import FIGURES, ROOT, table, write_result

DATA = ROOT / "data" / "external" / "cmapss"
SEED = 20261126
POSITIONS = (2, 4, 8)
WINDOWS = (500, 2000)
REPLICATES = 300
PRIOR_MEDIAN = 4.0
MASS = 0.90


def main() -> int:
    if not (DATA / "train_FD001.txt").exists():
        print("C-MAPSS data not found; run scripts/fetch_cmapss.py first. Skipping.")
        return 0
    lives = {s: load_lifetimes(DATA, s) for s in SUBSETS}
    truth = np.array([true_rate(lives[s]) for s in SUBSETS])
    data_rows = [
        [
            s,
            lives[s].size,
            int(lives[s].min()),
            round(float(lives[s].mean()), 1),
            int(lives[s].max()),
            round(float(lives[s].std() / lives[s].mean()), 3),
            round(float(truth[i]), 3),
        ]
        for i, s in enumerate(SUBSETS)
    ]
    rng = np.random.default_rng(SEED)
    prior = MeanCVHyperprior(mean_median=PRIOR_MEDIAN, mean_log_sd=1.0, cv_scale=1.0)
    a = (1 - MASS) / 2
    rows, disp_rows = [], []
    curves: dict[tuple[int, str], list[float]] = {}
    for window in WINDOWS:
        for k in POSITIONS:
            t = np.full(len(SUBSETS), k * window / 1000.0)
            err: dict[str, list[float]] = {
                "no pooling": [],
                "complete pooling": [],
                "hierarchical Bayes": [],
            }
            hit: dict[str, list[bool]] = {key: [] for key in err}
            all_counts = []
            for _ in range(REPLICATES):
                x = np.array([stationary_counts(rng, lives[s], k, window) for s in SUBSETS])
                all_counts.append(x)
                pooled = x.sum() / t.sum()
                plo, phi = garwood_interval(int(x.sum()), t.sum() * 1000, MASS)
                post = exact_posterior(x, t, prior, n=121)
                for i in range(len(SUBSETS)):
                    lo, hi = garwood_interval(int(x[i]), t[i] * 1000, MASS)
                    err["no pooling"].append((x[i] / t[i] - truth[i]) / truth[i])
                    hit["no pooling"].append(lo <= truth[i] <= hi)
                    err["complete pooling"].append((pooled - truth[i]) / truth[i])
                    hit["complete pooling"].append(plo <= truth[i] <= phi)
                    err["hierarchical Bayes"].append((post.unit_mean(i) - truth[i]) / truth[i])
                    c = post.unit_cdf(i, float(truth[i]))
                    hit["hierarchical Bayes"].append(a < c < 1 - a)
            counts = np.array(all_counts)
            mean_c = counts.mean(axis=0)
            disp_rows.append(
                [window, k]
                + [
                    f"{mean_c[i]:.1f} / {counts[:, i].var() / mean_c[i]:.2f}"
                    for i in range(len(SUBSETS))
                ]
            )
            for key in err:
                e = np.array(err[key])
                h = np.array(hit[key])
                rmse = float(np.sqrt(np.mean(e**2)))
                cov = float(h.mean())
                rows.append(
                    [
                        window,
                        k,
                        key,
                        f"{rmse:.3f} ± {np.std(e**2, ddof=1) / (2 * np.sqrt(e.size) * rmse):.3f}",
                        f"{cov:.3f} ± {np.sqrt(cov * (1 - cov) / h.size):.3f}",
                    ]
                )
                curves.setdefault((window, key), []).append(cov)

    plotting.use_style()
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(10, 3.6), sharey=True)
    colors = (plotting.FREQUENTIST_COLOR, plotting.NEUTRAL_COLOR, plotting.POOLED_COLOR)
    for ax, window in zip(axes, WINDOWS, strict=True):
        for key, color in zip(
            ("no pooling", "complete pooling", "hierarchical Bayes"), colors, strict=True
        ):
            ax.plot(POSITIONS, curves[(window, key)], "o-", color=color, label=key)
        ax.axhline(MASS, color=plotting.NEUTRAL_COLOR, lw=1, ls="--")
        ax.set_xticks(POSITIONS)
        ax.set_xlabel("engine positions per fleet")
        ax.set_title(f"window {window} cycles")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("coverage of the 90 % interval")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.savefig(FIGURES / "cmapss_coverage.png", metadata={"Software": None})
    plt.close(fig)

    write_result(
        "cmapss",
        "Extension: NASA C-MAPSS as four fleets with a known rate",
        [
            (
                "Data",
                "Training run-to-failure trajectories of NASA C-MAPSS (Saxena, Goebel, Simon "
                "and Eklund 2008), verified by SHA-256. Life is the last cycle of each "
                "training engine; the true rate is 1000 / mean life (renewal theorem), per 1000 "
                "cycles. FD004's file holds 249 engines, although its readme says 248.\n\n"
                + table(
                    [
                        "subset",
                        "engines",
                        "min life",
                        "mean life",
                        "max life",
                        "cv of life",
                        "true rate",
                    ],
                    data_rows,
                ),
            ),
            (
                "Setup",
                f"Each subset is a fleet of k engine positions, each a stationary renewal "
                "process with lives resampled from that subset, observed for a window of "
                f"cycles. The four subsets are the units of the hierarchical model, with "
                f"hyperprior μ ~ LogNormal(log {PRIOR_MEDIAN}, 1) (a 250-cycle mean life) and "
                f"cv ~ HalfNormal(1). Seeded Monte Carlo, {REPLICATES} replicates (seed {SEED}); "
                "RMSE and coverage pooled over the four subsets, ± one standard error.\n",
            ),
            (
                "Counts are under-dispersed",
                "Mean count / variance-to-mean ratio of each subset's window count (Poisson: "
                "ratio 1). Wear-out lives make counts more regular than Poisson.\n\n"
                + table(["window", "positions", *SUBSETS], disp_rows),
            ),
            (
                "Performance",
                table(
                    ["window (cycles)", "positions", "method", "RMSE / λ", "90 % coverage"], rows
                ),
            ),
        ],
        script="cmapss",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
