# Reliability statistics for small fleets: frequentist against Bayesian

*A methodology comparison on synthetic business-jet fleets with known failure rates, and on
NASA C-MAPSS lifetimes turned into fleets with a known renewal rate. Every number below is
taken from [`validation/results/`](../validation/results/) and can be regenerated with
`uv run python validation/run_all.py`.*

## Summary

The classical alert level of airline reliability programmes (FAA AC 120-17B; Transport Canada
AMA 571.101/1) was designed for fleets that log several events a month. A business-jet fleet
of five aircraft logs about one starter-generator removal every two years across the whole
fleet: 0.044 events a month.

1. **The classical method breaks, and it is possible to say exactly how.** On five aircraft,
   23 months in 24 have no events, and the 12-month baseline has a standard deviation of
   exactly zero in **59 %** of years, so the first event alerts. A year's 90 % interval is
   **7.3 times** as wide as the rate. The mean + 2σ rule's false-alarm rate is not the
   intended few percent: it drifts from **3 to 11 %** a month with fleet size.
2. **Borrowing from a related large fleet helps, if it is discounted.** A prior worth 10 000 FH
   of the large fleet's experience cuts the RMSE **2.6 to 5.7 times** on five aircraft. Counting
   the large fleet in full gives intervals that cover the true rate **0 %** of the time on four
   of six components. The data cannot warn you: the prior-data conflict check flags at most
   **2.7 %** of years.
3. **Pooling several operators of the same type is more robust than borrowing from a
   different fleet, if done fully Bayesian.** Plug-in empirical Bayes collapses to "all
   operators identical" in **40 to 71 %** of replicates, and its 90 % intervals cover **11 to
   50 %** of the time. Exact hierarchical Bayes on the same data covers **0.90 to 0.99**, with
   the lowest RMSE in every case, even with its hyperprior centred 20 % off.
4. **A Bayesian alert controls false alarms; it does not create information.** The rule
   "P(rate > target) > p*" is a count threshold with an exact false-alarm rate. On 25 aircraft
   a 3-month window detects a doubling more often than mean + kσ at matched false-alarm rates
   (**0.59 against 0.45** at about 3 % a month). On five aircraft no rule can do much better
   than alerting on any event, but only the Bayesian rule lets the false-alarm rate be chosen.
5. **The price of not knowing the rate can be stated in flight hours.** For a hidden failure
   with a 1007 FH test interval if its rate were known, a five-jet operator's 90 %-risk interval
   is **31 FH** after a year of its own data, **331 FH** with the large fleet's experience, and
   **403 FH** after twenty years alone.
6. **With a correct prior, no business-jet fleet is "big".** The borrowed estimate stays more
   than 10 % better than the MLE up to about **400 aircraft-years**. With a prior three times
   off, the gain is gone by 20 aircraft and the intervals are overconfident (**36 %** coverage
   on five aircraft).

## 1. The problem

A reliability programme estimates component failure rates from removals per 1000 flight
hours and raises an alert when a rate leaves its normal band. Both steps rely on many events
per period. The EASA guidance on small fleets (fewer than six aircraft of a type) says only
that alert levels "should be used carefully" and that data may be pooled. This report
measures what "carefully" means and what pooling buys.

## 2. Methods compared

| | estimate | interval | alert |
|---|---|---|---|
| classical | MLE | exact Garwood | mean + kσ of 12 monthly rates; Poisson 95 % count |
| borrowed prior | Gamma-Poisson posterior mean, power prior from a large fleet | equal-tailed credible | — |
| empirical Bayes | plug-in posterior from the negative-binomial marginal MLE | plug-in, Kass-Steffey | — |
| hierarchical Bayes | exact posterior by quadrature; PyMC for checking | exact | — |
| Bayesian alert | — | — | P(λ > target \| last w months) > p*, with p* from costs |

Each method is validated against closed forms or published values. NUREG/CR-6823 Table 6.4
(conjugate), Tables 8.2 and 8.3 (empirical Bayes) and Table 8.7 (WinBUGS hierarchical) are
reproduced, as are Transport Canada's alert-level example and NUREG's X² = 378.5.

## 3. Results

### The classical method (Phase 1, [`frequentist.md`](../validation/results/frequentist.md))

| aircraft | events / month | P(sd = 0) | 90 % width / λ | false alarms, 2σ | P(detect 2× in 3 months) |
|---|---|---|---|---|---|
| 5 | 0.044 | 0.592 | 7.29 | 0.042 | 0.23 |
| 25 | 0.219 | 0.072 | 2.42 | 0.108 | 0.45 |
| 200 | 1.75 | 0 | 0.77 | 0.066 | 0.67 |

Even 100 narrow-bodies (5 events a month) show 5.9 % false alarms at 2σ, above the 2.3 %
one-sided normal tail, because the baseline's mean and σ come from 12 points.

### Borrowing a prior (Phase 2, [`bayes_point.md`](../validation/results/bayes_point.md))

On five aircraft, a power prior counting the large fleet as 10 000 FH has RMSE 2.6 to 5.7 times
lower than the MLE's on every component, with conservative coverage. Full borrowing collapses
coverage to 0 on four of six components. Even where the two fleets' true rates are equal, a
2.4σ draw in the large fleet's data was enough.

### Pooling operators (Phase 3, [`empirical_bayes.md`](../validation/results/empirical_bayes.md), [`hierarchical.md`](../validation/results/hierarchical.md))

Operators of 2 to 12 business jets, one year, population cv 0.5:

| operators | method | RMSE / λ | 90 % coverage | width / λ |
|---|---|---|---|---|
| 4 | no pooling | 1.97 | 0.979 | 10.8 |
| 4 | empirical Bayes (Kass-Steffey) | 1.61 | 0.163 | 0.004 |
| 4 | hierarchical Bayes | 1.10 | 0.952 | 2.52 |
| 32 | no pooling | 1.63 | 0.979 | 7.49 |
| 32 | empirical Bayes (Kass-Steffey) | 0.98 | 0.501 | 1.09 |
| 32 | hierarchical Bayes | 0.92 | 0.897 | 1.81 |

NUTS reproduces the exact posteriors within Monte Carlo error. The centred parameterisation on
sparse data shows the hierarchical funnel (61 divergences), which the marginal form avoids
([`hierarchical_mcmc.md`](../validation/results/hierarchical_mcmc.md)).

### Alerts (Phase 4, [`alerts.md`](../validation/results/alerts.md))

| aircraft | rule | false alarms / month | P(detect 2× within a year) |
|---|---|---|---|
| 25 | mean + 2σ | 0.108 | 0.771 |
| 25 | Bayes, 3-month window, p* = 0.5 | 0.030 (exact 0.0290) | 0.588 |
| 100 | mean + 2σ | 0.079 | 0.905 |
| 100 | Bayes, 3-month window, p* = 0.5 | 0.054 | 0.951 |

### Decisions ([`decisions.md`](../validation/results/decisions.md))

With a missed month costing four investigations, p* = 0.2. On 25 aircraft that means alerting on
a single removal in three months, at 0.48 false alarms a month. That is the cost of
uncertainty made explicit, and an argument for the pooling above. Test intervals under
uncertainty are discussed in Summary item 5.

### When borrowing helps (Phase 5, [`simulation_study.md`](../validation/results/simulation_study.md))

| prior quality | RMSE ratio at 5 aircraft | coverage at 5 | converges at |
|---|---|---|---|
| good (×1.0) | 0.174 | 0.999 | 400 aircraft |
| biased (×1.5) | 0.279 | 1 | 200 aircraft |
| conflicting (×3.0) | 0.642 | 0.359 | 20 aircraft |

Hierarchical pooling across eight operators halves the RMSE on two-aircraft operators for
all three prior qualities, at 0.86 to 0.98 coverage.

### C-MAPSS ([`cmapss.md`](../validation/results/cmapss.md))

The four subsets give true renewal rates of 4.85, 4.84, 4.05 and 4.07 per 1000 cycles. Wear-out
makes window counts under-dispersed (variance-to-mean 0.06 to 0.22), so Poisson intervals
over-cover: misspecification in the safe direction. Partial pooling has the lowest RMSE on
short windows. On long windows the subsets' own data win, and complete pooling becomes
overconfident (0.73 coverage).

## 4. What this suggests for a small-fleet reliability programme

These are suggestions drawn from synthetic studies, not a procedure to adopt:

1. Report rates with **intervals**, not point estimates. On a small fleet the interval is the
   information.
2. Prefer **pooling across operators of the same type** (a manufacturer's fleet data) with a
   **fully Bayesian hierarchical model** to borrowing from a different fleet, and never plug in
   an empirical Bayes estimate without checking it against the full model.
3. When borrowing from a different fleet, **cap the borrowed exposure** (a few years of the
   small fleet's flying), because the data cannot detect a wrong prior.
4. State alerts as **probabilities with a cost-derived threshold**, and publish their exact
   false-alarm rates. Do not expect any rule to detect small changes on a five-aircraft fleet.
5. For hidden failures, choose test intervals from the **posterior**, not from a point estimate
   that is zero after a quiet year.

## 5. Limits

* **Synthetic data.** Component rates, utilisation, costs and targets are illustrative
  assumptions, labelled as such. The shapes of the results are general; crossover sizes and
  percentages depend on them.
* **Constant rates.** Removals are modelled as a homogeneous Poisson process within a period;
  real removals include wear-out, infant mortality, no-fault-found and seasonal effects.
  C-MAPSS probes one departure (wear-out), not all.
* **Exchangeability.** Pooling assumes operators are alike *a priori*. Known differences (route
  length, climate) belong in a regression, which is not built here.
* **Monte Carlo error.** Simulation-based tables report standard errors or are seeded and
  bounded. Exact tables (Phases 1, 2, 5 single fleet, decisions) carry no simulation error.
* **Not a certification method.** Nothing here replaces an approved reliability programme or
  engineering judgement. An alert, classical or Bayesian, is a prompt to investigate (AC
  120-17B §4.2.1).
* **Novelty.** The methods are established; see [`related-work.md`](related-work.md) for what
  is and is not new, and for the limits of the literature search.
