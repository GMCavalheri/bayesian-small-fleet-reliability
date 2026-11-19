# PLAN.md — EJD-01: Bayesian Reliability Analysis for Small Fleets

## Overview
A reliability engineering project addressing a problem the classical statistical alert-level methodology (used for large commercial fleets) handles poorly: estimating component failure rates with statistically sparse data, as is typical for executive jet fleets (Phenom, Praetor, Legacy-class), which are orders of magnitude smaller than a commercial narrow-body fleet.

## Objective
Build a Bayesian reliability estimation framework that produces credible failure rate estimates and alert thresholds for small fleets, using informative priors and partial pooling across similar components/fleets, and compare it against the classical frequentist approach used in large-fleet reliability programs.

## Why This Project
The Poisson-based alert-level methodology underlying commercial airline reliability programs implicitly assumes enough events to have statistical power. Executive jet fleets rarely have that luxury — a single fleet might log only a handful of failures per component per year. This is a genuinely underserved technical problem in aviation reliability engineering, and Bayesian methods (which formally incorporate prior knowledge and can borrow strength across related components) are the standard statistical answer — but they are rarely implemented by reliability engineers who only know the classical approach.

## Knowledge Covered
1. **Frequentist Baseline**: reimplement the classical Poisson-based alert-level calculation on a simulated small-fleet dataset, and show where it breaks down (wide, unstable confidence intervals; alert thresholds that rarely trigger meaningfully with so few events).
2. **Bayesian Point Estimation**: implement Bayesian estimation of failure rate (Gamma-Poisson conjugate model) using informative priors derived from a related large-fleet component (e.g., a shared engine or avionics component also used on a commercial platform), producing tighter, more defensible credible intervals than the frequentist approach on sparse data.
3. **Hierarchical/Partial Pooling Model**: implement a hierarchical Bayesian model across multiple related components or multiple small fleets simultaneously, allowing each component's rate estimate to "borrow strength" from the group while still allowing individual deviation — this is the core statistical contribution of the project.
4. **Bayesian Alert Thresholds**: reformulate the alert-level concept (from the classical reliability program) in Bayesian terms — e.g., posterior probability that the true failure rate exceeds a design target — as a more defensible alternative to a fixed frequentist control limit for sparse data.
5. **Simulation Study**: systematically compare frequentist vs. Bayesian approaches across simulated fleets of varying size, quantifying at what fleet size the two approaches converge and where Bayesian methods provide the most benefit.

## Prerequisites / Background
- The classical reliability alert-level methodology already used in the Fleet Reliability Dashboard project (Poisson-based control limits)
- Bayesian statistics fundamentals: priors, likelihood, posterior, credible intervals
- Hierarchical/multilevel modeling concepts (partial pooling)

## Data / Dataset
- **Synthetic small-fleet data (primary)**: simulated failure/removal histories for fleets of different sizes (e.g., 5, 15, 50 aircraft), generated from known true failure rates so estimation error can be measured directly.
- **Prior sources**: a synthetic "large-fleet" population sharing components with the small fleet (e.g., AVR-02's simulated commercial fleet), standing in for the related-fleet data a real operator or OEM would use.
- **Optional validation case**: NASA C-MAPSS sub-datasets (via AVR-01) as groups for the hierarchical model, with known ground truth.

## Tech Stack
- Python, NumPy, SciPy
- PyMC or NumPyro (probabilistic programming for hierarchical Bayesian models)
- Matplotlib/Plotly for posterior distributions, credible interval comparisons
- Pytest for validation against known conjugate-prior closed-form results

## Repository Structure
```
bayesian-small-fleet-reliability/
  src/
    frequentist_baseline/
    bayesian_point_estimation/
    hierarchical_model/
    bayesian_alert_thresholds/
    simulation_study/
  validation/
  notebooks/
  docs/
  tests/
```

## Execution Plan
- [x] **Phase 1 — Frequentist baseline and failure mode demonstration**: implement the classical Poisson alert-level method on synthetic small-fleet data, document its instability as fleet size shrinks.
- [x] **Phase 2 — Bayesian point estimation**: implement Gamma-Poisson conjugate Bayesian estimation with informative priors, validate against closed-form conjugate posterior results.
- [x] **Phase 3 — Hierarchical model**: implement partial pooling across multiple components/fleets using PyMC/NumPyro, validate convergence diagnostics (R-hat, effective sample size).
- [x] **Phase 4 — Bayesian alert thresholds**: reformulate alert logic in terms of posterior probability of exceedance, compare trigger behavior against the classical method.
- [x] **Phase 5 — Simulation study**: run a systematic comparison across simulated fleet sizes, produce a clear "when does Bayesian pooling help" result.
- [x] **Phase 6 — Documentation and publishing**: written methodology comparison, visualizations of posterior distributions and simulation results.

## Validation Strategy
- Bayesian point estimates validated against closed-form Gamma-Poisson conjugate posterior formulas for simple cases.
- Hierarchical model validated via posterior predictive checks and standard MCMC convergence diagnostics.
- Simulation study designed with known ground-truth failure rates so estimation error can be measured directly.

## Deliverables
- Public repository with a documented Bayesian reliability estimation framework
- A simulation-based comparison report: frequentist vs. Bayesian approaches across fleet sizes
- Visualizations: posterior distributions, credible interval width vs. fleet size, alert threshold comparison

## Portfolio Differentiators
- Addresses a real, underserved gap in aviation reliability practice — most reliability engineers only know the classical large-fleet methodology, and small-fleet/business aviation reliability is a genuinely under-published problem.
- Statistically rigorous (proper Bayesian hierarchical modeling, not just "using more data") — a strong technical signal for a product reliability role in business aviation.

## Possible Extensions
- Apply the hierarchical model across the RUL Prediction project's C-MAPSS sub-datasets as an additional validation case with known ground truth
- Extend to a full Bayesian decision-theoretic framework for maintenance interval setting under uncertainty
