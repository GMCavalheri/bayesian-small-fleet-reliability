# CLAUDE.md

Bayesian reliability for small fleets. The classical alert-level method of airline reliability
programs (a control limit on monthly failure rates per 1000 flight hours) assumes a fleet large
enough to log many events; a small business-jet fleet logs a handful per component per year.
This project shows where the classical method breaks, then estimates failure rates and alert
thresholds from sparse data with Bayesian methods (an informative prior from a related large
fleet, empirical Bayes, hierarchical partial pooling, alerts as posterior probabilities,
expected-loss decisions), and measures by simulation at what fleet size the approaches
converge.
Plan: `EJD-01-bayesian-small-fleet-reliability-PLAN.md` (tick the phase checkboxes as phases
land). The classical method is the one AVR-02 (`GMCavalheri/Aircraft-Fleet-Reliability-Dashboard`)
studies; keep the units (rates per 1000 FH, monthly periods) consistent with it.

## Commands
- `uv sync` — install (Python ≥ 3.11; CI runs 3.11, 3.12 and 3.13)
- `uv run pytest --cov` — tests with the 90 % coverage gate (`-m "not slow"` skips MCMC tests)
- `uv run ruff check .` and `uv run ruff format --check .` — lint/format. Ruff also formats
  the ```python blocks inside Markdown, so run it after editing docs too
- `uv run mypy` — strict type check of `src/`
- `uv run python validation/<script>.py` — regenerate one phase's tables
  (`validation/results/`) and figures (`docs/figures/`); `validation/run_all.py` runs them all
- `uv run python notebooks/build_notebooks.py` — regenerate and execute the notebooks
- `uv run python scripts/fetch_cmapss.py` — download NASA C-MAPSS (checksummed, never committed)

## Layout
- `src/smallfleet/` — single package; subpackages mirror the plan:
  `data_generation` (synthetic small and large fleets of known true rates),
  `frequentist_baseline` (MLE, Garwood interval, classical alert level),
  `bayesian_point_estimation` (Gamma-Poisson conjugate model, priors from a large fleet,
  power prior, prior-data conflict), `empirical_bayes`, `hierarchical_model` (PyMC model and
  the exact quadrature oracle), `bayesian_alert_thresholds`, `decision_theory`,
  `simulation_study`, `cmapss`; plus `plotting.py`.
- `src/smallfleet/data/` — parameter tables (CSV); every value is labelled an assumption or
  carries its source.
- `tests/` — pytest + Hypothesis. `validation/` — one script per phase, `results/` tables.
- `docs/` — theory note per module, `scope.md`, `related-work.md`, `report.md`;
  `docs/lessons/` is the course (keep its numbers in sync with `validation/results/`).
  `notebooks/` — one executed demo per lesson.

## Conventions
- Exposure is in flight hours (FH); rates are failures per 1000 FH; the reporting period is a
  calendar month, as in airline reliability programs. Failures are a Poisson process with
  constant rate λ over a period (a stated assumption).
- Gamma distributions use the shape-rate form, Gamma(α, β) with mean α/β, which makes the
  Poisson update α + y, β + T. Note that `scipy.stats.gamma` takes `scale = 1/β`.
- Simulations take a seeded `numpy.random.Generator`; PyMC takes a fixed `random_seed`,
  `cores=1` and fixed chains/draws.
- Exact results (closed forms, quadrature) are compared with tolerance 1e-10 or better; Monte
  Carlo and MCMC against their own standard errors (state the multiple).
- MCMC-derived numbers in `validation/results/` are rounded to 2 significant figures so they
  reproduce across machines; headline numbers come from the exact oracle where it applies.

## Scope (public repo)
- Synthetic fleets and public methodology only (FAA AC 120-17B, NUREG/CR-6823, the Bayesian
  reliability textbooks). Paid standards are named, never quoted. NASA C-MAPSS is public
  data, downloaded by the user and never committed. See `docs/scope.md`.
- Component rates are illustrative assumptions, never presented as field data of any operator
  or manufacturer. Fleets are generic ("light / midsize / large-cabin business jet").
- Never mention the author's employer, day job or a target employer in any file, commit
  message or doc.

## Docs (GitHub-rendered math)
- Display equations go in ```` ```math ```` fences (no Markdown processing inside).
- Inline `$...$` is processed as Markdown first: no `\,` or `\;`, no `\\` row breaks, and no
  `_` right after `}` `)` `]`. `tests/test_docs.py` guards this.

## Testing rules
- Every test asserts against a closed-form result, an independent oracle (numerical
  integration, the exact quadrature posterior, brute-force optimisation, Monte Carlo) or a
  cited published value; name the source in a comment.
- Never loosen a tolerance to make a failing test pass without explaining why.
- Properties that must hold for every parameter set (the posterior mean lies between the prior
  mean and the MLE, exceedance probability rises with the count, EVSI ≥ 0) are Hypothesis
  tests.

## Workflow
- Commit small logical changes directly to `main`; push after ruff, mypy and pytest pass,
  each run as its own command (no pipes: a pipe hides the exit code). Stage explicit paths.
- Each phase lands as code + tests, then validation results, then its lesson in
  `docs/lessons/` as a separate commit. Confirm CI is green after each push.
