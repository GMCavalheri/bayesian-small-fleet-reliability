# Scope and modelling boundaries

This repository is a **reliability statistics methodology and software** project. It asks
one question: how should a component's failure rate and alert threshold be estimated when a
fleet is too small for the classical method of airline reliability programmes? Everything in
it is meant to be safe to publish, and these rules keep it that way.

## What is modelled

* **Synthetic fleets.** Fleets of 3 to 200 generic business jets ("light / midsize /
  large-cabin business jet", never a named type) and a generic narrow-body fleet that
  supplies prior information. Their utilisation and component removal rates are in
  [`fleet_parameters.csv`](../src/smallfleet/data/fleet_parameters.csv) and
  [`component_parameters.csv`](../src/smallfleet/data/component_parameters.csv). Every value
  is an **illustrative assumption**, labelled as such, chosen to be of a plausible order of
  magnitude. None is field data of any operator, manufacturer or programme, and none should
  be read as one.
* **Known truth.** Because the data are simulated from known rates, every estimate's error,
  every interval's coverage and every alert rule's false-alarm and detection rates can be
  measured directly. That is the point of the synthetic design.
* **Public methodology.** The classical method follows FAA AC 120-17B and Transport Canada
  AMA 571.101/1, Appendix A. The Bayesian methods follow the public textbook and NUREG
  literature (NUREG/CR-6823, Hamada et al., Kelly & Smith, Gelman et al.). Paid standards are
  named, never quoted.
* **One public dataset.** NASA's C-MAPSS turbofan degradation data (Lesson 8) is a public,
  simulated dataset. It is downloaded by the user with `scripts/fetch_cmapss.py`, checked
  against checksums, and never committed.

## What is not modelled

* No real operator's reliability data, alert levels, maintenance programme or fleet.
* No airworthiness or certification judgement. An alert, classical or Bayesian, is a prompt
  to investigate (AC 120-17B, §4.2.1), not a statement about airworthiness, and nothing here
  replaces an approved reliability programme or engineering judgement.
* No claim about any manufacturer's products or their reliability.

## Why this is the right boundary

The statistical question (how much a few events can tell you, and how much a related fleet
can add) depends only on the expected number of events, the prior information and the
decision rule. It does not depend on whose aircraft the numbers come from. Synthetic data
with known truth answer it better than real data could, because real data never reveal the
true rate.
