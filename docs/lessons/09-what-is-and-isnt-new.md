# Lesson 9 · What is (and isn't) new

> A course that teaches methods should also teach how to place work among what others have
> done. This last lesson reads the literature behind the previous eight, separates what
> this project borrowed from what it adds, and practises the skill that matters most when
> presenting applied work: saying how new something is, no more and no less.

**Note:** [`docs/related-work.md`](../related-work.md) (every reference checked against its
source) · **Previous:** [Lesson 8](08-a-dataset-with-real-ground-truth.md)

## Learning objectives

By the end of this lesson you should be able to:

1. Place each method of the course in its literature: who introduced it, and where it is
   standard.
2. Tell the difference between a new method, a new application and a new measurement.
3. Write a novelty statement whose claims are each backed by a result, and whose limits
   are stated.
4. Plan a systematic search that could overturn it.

---

## 1. Three kinds of contribution

Applied statistical work usually contributes one of three things:

* **a new method:** an estimator, model or algorithm that did not exist;
* **a new application:** a known method used where it had not been used;
* **a new measurement:** a careful, quantified answer to a question that people had
  answered by rule of thumb.

This project has **no new method**. Every model in it is decades old, and the tests prove
it by reproducing the published examples digit for digit. Its claims are of the third kind,
and partly of the second.

## 2. Where each piece comes from

| lesson | method | origin | where it is standard |
|---|---|---|---|
| 1 | mean + kσ and Poisson alert levels | airline reliability programmes | FAA AC 120-17B; TC AMA 571.101/1 |
| 1 | exact Poisson interval | Garwood (1936) | every statistics textbook |
| 2 | Gamma-Poisson, Jeffreys prior | Bayes, Laplace; Jeffreys (1946) | NUREG/CR-6823 §6.2 |
| 2 | power prior | Ibrahim & Chen (2000) | clinical trials, historical borrowing |
| 3 | parametric empirical Bayes, Kass-Steffey | Efron & Morris; Kass & Steffey (1989) | NUREG/CR-6823 §8.2 |
| 4 | hierarchical Bayes, MCMC | Gelman et al.; Hoffman & Gelman (2014) | NUREG/CR-6823 §8.3 |
| 4 | posterior predictive checks, SBC | Gelman, Meng & Stern (1996); Cook et al. (2006) | Bayesian workflow |
| 5 | posterior-probability alerts | Bayesian SPC: Bourazas et al. (2021) | quality control of short runs |
| 6 | expected loss, EVSI | Raiffa & Schlaifer (1961); Berger (1985) | decision analysis |
| 6 | failure-finding interval | Moubray (1997) | RCM / MSG-3 practice |
| 8 | renewal theory | Cox (1962); Ross (1996) | reliability theory |

The table reads as a warning. Anyone who presents this work as "a Bayesian method for small
fleets" is claiming a column it does not own.

## 3. The closest work

The search (related-work note) found two pieces that a reviewer would bring up first:

* **Arnaldo Valdés et al. (2018, *Safety Science*)** fit hierarchical Gamma-Poisson and
  Beta-binomial models to aircraft **incident** rates across fleets. It is the same statistics,
  in aviation. Ours differs in its target (maintenance alert levels and component removal
  rates), its question (when does the classical rule break, and what replaces it), and its
  evidence (exact repeated-sampling performance against known truth).
* **Bourazas, Kiagias & Tsiamyrtzis (2021, *Journal of Quality Technology*)** monitor Poisson
  counts with Bayesian predictive control charts and power priors, for short runs. It is the
  same idea as Lesson 5's alert, in industrial process control. Ours adds the paired
  comparison with the airline alert level at matched false-alarm rates, and the information
  limit on very small fleets.

Two more are close in motivation. Küçüker & Yet (2023) use a Bayesian network to combine
supplier estimates with scarce operator data. Lewis-Beck, Tian & Meeker (2021) pool
subpopulations hierarchically to predict failures from few events.

## 4. The novelty statement

> *Using synthetic business-jet fleets with known failure rates, and NASA C-MAPSS lifetimes
> turned into fleets with a known renewal rate, this project quantifies how the classical
> alert level of airline reliability programmes degrades on small fleets. Twelve-month
> baselines have zero spread in 59 % of years on five aircraft, and false-alarm rates drift
> from 3 to 11 % with fleet size. It compares that level, at matched false-alarm rates,
> with a posterior-exceedance alert whose false-alarm rate is exact. It shows that
> empirical Bayes pooling of a few sparse operators is overconfident (11 to 50 % coverage),
> where exact hierarchical Bayes is not (0.90 to 0.99). All results come from open,
> tested software that reproduces the published NUREG/CR-6823 examples. We did not find
> these measurements published for small business-aviation fleets; a systematic search of
> the reliability journals and conference proceedings remains to be done.*

Notice three things:

1. **Every number is a result** in `validation/results/` and can be regenerated.
2. **The claim is about measurement and application**, not method.
3. **"We did not find" is not "nobody did".** The last sentence says how the claim could
   fail.

## 5. How the search could overturn it

A systematic search would cover:

* *Reliability Engineering & System Safety* and *IEEE Transactions on Reliability*, by
  keywords (alert level, reliability programme, small fleet, sparse, Gamma-Poisson,
  hierarchical, empirical Bayes) and by forward citations of NUREG/CR-6823 §8 and of Arnaldo
  Valdés et al. (2018);
* RAMS and PSAM proceedings, and the SAE and ATA maintenance-reliability literature;
* military small-fleet reliability reports (mission-capable rates and removal rates of
  low-density fleets), which are public in part;
* the Bayesian SPC literature after Bourazas et al. (2021), for applications to aircraft.

If any of these already reports items 1, 2 or 4 of the novelty statement, the statement
shrinks to "an open, tested reproduction".

## 6. Worked example: rewriting an overclaim

*Overclaim:* "We propose a new Bayesian framework that solves the small-fleet reliability
problem."

* "propose a new … framework": the methods are not new (Section 2);
* "solves": Lesson 5 shows that no rule detects a doubling much better than "alert on an
  event" on five aircraft;
* "the small-fleet reliability problem": the work covers constant-rate component removals,
  not wear-out life limits or certification.

*Rewrite:* "We measure where the classical alert level fails on small fleets, and how much
established Bayesian methods (borrowed priors, hierarchical pooling, posterior-probability
alerts) recover, with exact results against known truth."

## Exercises

1. Classify each as a new method, application or measurement: (a) the exact quadrature
   oracle; (b) the zero-spread rate of empirical Bayes fits; (c) using C-MAPSS lifetimes as
   renewal fleets.
2. Arnaldo Valdés et al. (2018) and this project use the same model family. Name two
   questions a reviewer could ask to tell the contributions apart.
3. Which single result in the course would most need independent replication before being
   presented to a reliability board, and why?
4. Rewrite "Bayesian alerts outperform classical alerts" so that it is true on the evidence of
   Lesson 5.

### Solutions

1. (a) An implementation of a known computation: quadrature over two hyperparameters. It is
   standard, and NUREG uses MCMC only for convenience. As software it is a contribution; as
   statistics it is not. (b) A measurement. (c) A new application: a known dataset used for a
   new question.
2. "What decision does your model feed?" (safety incident monitoring, or maintenance alerting
   and test intervals). "Against what truth are your estimates scored?" (real data with an
   unknown truth, or simulated fleets with a known one).
3. The fleet-size crossover points (Lesson 7). They depend on the illustrative rates,
   utilisation and prior weight. Their shape is general; the numbers should be redone with
   an operator's real utilisation and an OEM's real prior before anyone sets policy on them.
4. "At matched false-alarm rates, a 3-month posterior-exceedance alert detected a doubling of
   the rate more often than mean + kσ on 25 aircraft (0.59 against 0.45 at about 3 % false
   alarms per month). On 5 aircraft neither rule could do much better than alerting on any
   event, but the Bayesian rule's false-alarm rate could be set in advance."

## References

See [`docs/related-work.md`](../related-work.md) for the full, checked list. Also:

* F. Garwood, "Fiducial limits for the Poisson distribution," *Biometrika* 28(3/4), 1936.
* H. Jeffreys, "An invariant form for the prior probability in estimation problems,"
  *Proceedings of the Royal Society A* 186(1007), 1946.
* B. Efron and C. Morris, "Data analysis using Stein's estimator and its generalizations,"
  *Journal of the American Statistical Association* 70(350), 1975.
* A. Gelman, A. Vehtari, D. Simpson et al., "Bayesian workflow," arXiv:2011.01808, 2020.
