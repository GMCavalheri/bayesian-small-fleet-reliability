"""Generate and execute the demo notebooks, one per lesson, from the cell sources below.

    uv run python notebooks/build_notebooks.py           # write and execute all notebooks
    uv run python notebooks/build_notebooks.py --check   # execute in memory only (CI)

Keeping the sources here, instead of hand-editing .ipynb JSON, makes the notebooks reviewable
in diffs. Every computation is exact or seeded and runs in seconds; the headline numbers live
in ``validation/results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """\
import matplotlib.pyplot as plt
import numpy as np

from smallfleet import plotting

plotting.use_style()
rng = np.random.default_rng(1)"""

Cell = tuple[str, str]


def intro(n: int, title: str, lesson: str, text: str) -> Cell:
    return ("md", f"# {n} · {title}\n\n{text} See [Lesson {n}](../docs/lessons/{lesson}).")


NOTEBOOKS: dict[str, list[Cell]] = {
    "01_the_alert_level_that_needs_a_big_fleet.ipynb": [
        intro(
            1,
            "The alert level that needs a big fleet",
            "01-the-alert-level-that-needs-a-big-fleet.md",
            "Simulate a small fleet, compute the classical alert level, and see how often it is degenerate.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from smallfleet.data_generation import simulate_history
from smallfleet.frequentist_baseline import (
    degenerate_baseline_probability,
    garwood_interval,
    mean_sd_alert_level,
    monthly_rates,
)

h = simulate_history(rng, rate=0.25, n_aircraft=5, fh_per_aircraft_month=35.0, months=24)
base = monthly_rates(h.counts[:12], h.exposure_fh[:12])
print("baseline counts:", h.counts[:12])
print("mean + 2 sd level:", mean_sd_alert_level(base, 2.0))
print("90 % interval, 24 months:", garwood_interval(h.total_count, h.total_exposure_fh, 0.90))
print("P(baseline sd = 0):", degenerate_baseline_probability(0.25, 175.0))""",
        ),
        (
            "code",
            """\
sizes = np.array([3, 5, 10, 25, 50, 100, 200])
p = [degenerate_baseline_probability(0.25, n * 35.0) for n in sizes]
fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogx(sizes, p, "o-", color=plotting.FREQUENTIST_COLOR)
ax.set_xlabel("aircraft")
ax.set_ylabel("P(12-month baseline has sd = 0)");""",
        ),
    ],
    "02_borrowing_a_prior_from_a_bigger_fleet.ipynb": [
        intro(
            2,
            "Borrowing a prior from a bigger fleet",
            "02-borrowing-a-prior-from-a-bigger-fleet.md",
            "The conjugate update, the power prior and the bias-variance cost of borrowing.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from smallfleet.bayesian_point_estimation import (
    bayes_estimator,
    exact_performance,
    frequentist_estimator,
    jeffreys,
    power_prior,
)

for eq in (0.0, 10_000.0, 900_000.0):
    prior = jeffreys() if eq == 0 else power_prior(171, 900_000.0, eq)
    post = prior.update(1, 2100.0)
    print(f"equivalent {eq:>9,.0f} FH: mean {post.mean():.3f}, 90 % {post.interval()}")""",
        ),
        (
            "code",
            """\
eqs = np.geomspace(500, 900_000, 30)
rmse = [exact_performance(bayes_estimator(power_prior(171, 900_000.0, e)), 0.25, 2100.0).rmse for e in eqs]
mle = exact_performance(frequentist_estimator(), 0.25, 2100.0).rmse
fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogx(eqs, np.array(rmse) / 0.25, color=plotting.BAYES_COLOR)
ax.axhline(mle / 0.25, color=plotting.FREQUENTIST_COLOR, ls=":")
ax.set_xlabel("large-fleet exposure counted (FH)")
ax.set_ylabel("RMSE / true rate");""",
        ),
    ],
    "03_letting_the_data_choose_the_prior.ipynb": [
        intro(
            3,
            "Letting the data choose the prior",
            "03-letting-the-data-choose-the-prior.md",
            "Empirical Bayes on NUREG/CR-6823 Example 8.1, and its collapse on sparse operators.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
import sys

sys.path.insert(0, "../tests")
from nureg import example_8_1

from smallfleet.empirical_bayes import fit_marginal_mle, kass_steffey_posteriors

names, x, t = example_8_1()
fit = fit_marginal_mle(x, t)
print(f"population: Gamma({fit.alpha:.3f}, {fit.beta:.3f})")
post = kass_steffey_posteriors(fit, x, t)[names.index("Callaway")]
print("Callaway:", round(post.mean(), 2), [round(v, 2) for v in post.interval()])""",
        ),
        (
            "code",
            """\
sparse = fit_marginal_mle(np.array([0, 1, 3]), np.array([0.84, 1.68, 2.52]))
print("three sparse operators -> pooled:", sparse.pooled, "mean", round(sparse.mean, 3))""",
        ),
    ],
    "04_partial_pooling.ipynb": [
        intro(
            4,
            "Partial pooling",
            "04-partial-pooling.md",
            "The exact hierarchical posterior, and NUTS checked against it.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from smallfleet.hierarchical_model import MeanCVHyperprior, exact_posterior
from smallfleet.hierarchical_model.model import sample

x = np.array([0, 1, 0, 2, 1, 4])
t = np.array([2, 3, 4, 6, 8, 12]) * 35 * 12 / 1000
prior = MeanCVHyperprior(mean_median=0.2, mean_log_sd=1.0, cv_scale=1.0)
post = exact_posterior(x, t, prior)
for i in range(x.size):
    print(f"operator {i + 1}: {post.unit_mean(i):.3f} {post.unit_interval(i)}")""",
        ),
        (
            "code",
            """\
r = sample(x, t, prior, marginal=True, draws=500, tune=500, seed=1)
print("max R-hat", round(r.rhat_max, 3), "| min ESS", round(r.ess_bulk_min), "| divergences", r.divergences)
print("operator 1, NUTS:", round(r.unit_mean(0), 3), "exact:", round(post.unit_mean(0), 3))""",
        ),
    ],
    "05_an_alert_as_a_probability.ipynb": [
        intro(
            5,
            "An alert as a probability",
            "05-an-alert-as-a-probability.md",
            "The posterior-exceedance alert, its count threshold and its exact false-alarm rate.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from smallfleet.bayesian_alert_thresholds import (
    alert_count_threshold,
    exceedance_probability,
    window_false_alarm_rate,
)
from smallfleet.bayesian_point_estimation import GammaRate

prior = GammaRate(2.5, 10.0)
for c in range(4):
    print(c, "events in 6 months:", round(exceedance_probability(prior, c, 1050.0, 0.375), 3))
print("alert count:", alert_count_threshold(prior, 1050.0, 0.375, 0.5))
print("exact false alarms per month:", window_false_alarm_rate(prior, 0.25, 175.0, 6, 0.375, 0.5))""",
        ),
    ],
    "06_deciding_under_uncertainty.ipynb": [
        intro(
            6,
            "Deciding under uncertainty",
            "06-deciding-under-uncertainty.md",
            "The alert threshold from costs, the value of waiting, and a test interval under uncertainty.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.decision_theory import (
    AlertCosts,
    bayes_threshold,
    interval_for_expected_unavailability,
    interval_for_known_rate,
    interval_for_risk,
    value_of_waiting,
)

costs = AlertCosts(1.0, 4.0, 0.5)
print("threshold:", bayes_threshold(costs))
print(value_of_waiting(GammaRate(2.5, 10.0).update(1, 2625.0), 0.375, 875.0, costs))
device = GammaRate(1.5, 50.0).update(0, 6300.0)
print("known rate:", round(interval_for_known_rate(0.02, 0.01)), "FH")
print("expected U:", round(interval_for_expected_unavailability(device, 0.01)), "FH")
print("90 % risk:", round(interval_for_risk(device, 0.01, 0.9)), "FH")""",
        ),
    ],
    "07_how_big_is_a_big_fleet.ipynb": [
        intro(
            7,
            "How big is a big fleet?",
            "07-how-big-is-a-big-fleet.md",
            "The exact RMSE ratio against fleet size for good, biased and conflicting priors.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from smallfleet.simulation_study import single_fleet_point

sizes = [2, 5, 12, 30, 80, 200, 400]
fig, ax = plt.subplots(figsize=(5, 3))
for ratio, color in zip((1.0, 1.5, 3.0), plotting.SERIES_COLORS[1:4]):
    pts = [single_fleet_point(n, 0.2 * ratio, 0.2, 35.0, 12, 10_000.0, 900_000.0) for n in sizes]
    r = [p.methods["borrowed prior"].rmse / p.methods["MLE + Garwood"].rmse for p in pts]
    ax.semilogx(sizes, r, "o-", color=color, label=f"true / large = {ratio}")
ax.axhline(1, color=plotting.NEUTRAL_COLOR, lw=1)
ax.set_xlabel("aircraft")
ax.set_ylabel("RMSE ratio to the MLE")
ax.legend(fontsize=8);""",
        ),
    ],
    "08_a_dataset_with_real_ground_truth.ipynb": [
        intro(
            8,
            "A dataset with real ground truth",
            "08-a-dataset-with-real-ground-truth.md",
            "Renewal counts from C-MAPSS lives (or illustrative lives when the data are not downloaded).",
        ),
        ("code", SETUP),
        (
            "code",
            """\
from pathlib import Path

from smallfleet.cmapss import load_lifetimes, stationary_counts, true_rate

data = Path("../data/external/cmapss")
if (data / "train_FD001.txt").exists():
    lives = load_lifetimes(data, "FD001")
    print("C-MAPSS FD001:", lives.size, "engines")
else:
    lives = rng.normal(206, 46, 100).round().clip(128).astype(int)
    print("C-MAPSS not downloaded (run scripts/fetch_cmapss.py); using illustrative lives")
counts = np.array([stationary_counts(rng, lives, 4, 2000) for _ in range(500)])
print("true rate:", round(true_rate(lives), 3), "| mean count:", counts.mean(), "| expected:", 4 * 2000 / lives.mean())
print("variance / mean:", round(counts.var() / counts.mean(), 3), "(Poisson: 1)")""",
        ),
    ],
    "09_what_is_and_isnt_new.ipynb": [
        intro(
            9,
            "What is (and isn't) new",
            "09-what-is-and-isnt-new.md",
            "Reproducing the published examples that anchor the claims.",
        ),
        ("code", SETUP),
        (
            "code",
            """\
import sys

sys.path.insert(0, "../tests")
from nureg import example_8_1

from smallfleet.bayesian_point_estimation import GammaRate
from smallfleet.hierarchical_model import AlphaBetaHyperprior, exact_posterior

post = GammaRate(1.53, 10.63).update(1, 4890.0)  # NUREG/CR-6823 Example 6.1, Table 6.4
print("Table 6.4, industry prior:", round(post.mean(), 2), [round(v, 3) for v in post.interval()])
names, x, t = example_8_1()
hb = exact_posterior(x, t, AlphaBetaHyperprior())
i = names.index("Callaway")
print("Table 8.7, Callaway:", round(hb.unit_mean(i), 2), [round(v, 2) for v in hb.unit_interval(i)])""",
        ),
    ],
}


def build(name: str, cells: list[Cell]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"{Path(name).stem}-{i}"
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        record_timing=False,  # no timestamps: rebuilds are byte-identical
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="execute without writing")
    ap.add_argument("names", nargs="*", help="notebooks to build (default: all)")
    args = ap.parse_args()
    for name in args.names or list(NOTEBOOKS):
        print(f"== {name}", flush=True)
        nb = build(name, NOTEBOOKS[name])
        if not args.check:
            nbformat.write(nb, HERE / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
