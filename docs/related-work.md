# Related work

What this project builds on, what is close to it, and what (narrowly) it adds. Every
reference below was checked against its source (publisher page, Crossref record or the
document itself) in November 2026. Where a document could only be reached through a
secondary copy, that is stated.

## 1. The classical alert level

* **FAA, Advisory Circular 120-17B** (2018), *Reliability Program Methods — Standards for
  Determining Time Limitations*. §4.1 asks for alert values based on "generally accepted
  statistical methods, such as standard deviation or the Poisson distribution", and for
  procedures to monitor new fleet types until enough experience exists. §4.2.1 defines an
  alert level as a way to identify increases beyond normal variation, not an airworthiness
  limit.
* **Transport Canada, AMA 571.101/1, Appendix A**, *Calculation of Alert Levels*. Worked
  examples of the mean-ratio, mean + 2σ, mean + 3σ and Poisson-chart methods. Example 4(a) is
  reproduced by this project's tests. Example 4(b)'s chart reading does not match the 95 %
  rule it states (Lesson 1).
* **EASA, Appendix I to AMC M.A.302, §6.2** ("Applicability, small fleets"), consulted through
  the Bahrain CAA checklist ALD/AIR/F170 (Rev. 5, 2023), which itemises it. A small fleet is
  fewer than 6 aircraft of a type; alert levels "should be used carefully"; pooling of data is
  allowed under conditions. (The UK CAA's online copy requires a login.)
* **J. Amborski**, "Calculation of alert levels for reliability," *European Conference on
  Modelling and Simulation (ECMS)*, 2009. Describes mean + standard deviation and Weibull-based
  alert levels, and notes that alert levels reflect the experience of many aircraft of the
  same type. It does not treat sparse data or Bayesian methods.

Together these are the method of AVR-02 and Lesson 1. None of them gives a statistical
treatment of the small-fleet case beyond "use carefully" and "pool".

## 2. Bayesian estimation of failure rates

* **C. L. Atwood et al.**, *Handbook of Parameter Estimation for Probabilistic Risk Assessment*,
  NUREG/CR-6823, U.S. NRC, 2003. The reference for Gamma-Poisson estimation, Jeffreys and
  constrained noninformative priors, parametric empirical Bayes with the Kass-Steffey
  adjustment, and hierarchical Bayes by MCMC. Its Examples 6.1 and 8.1 (66 plants) are
  reproduced exactly by this project's tests, including the WinBUGS hierarchical results.
  The printed derivative ∂E/∂α̂ of §8.2.4.1 is inconsistent with its own Table 8.3 (Lesson 3).
* **C. L. Atwood**, "Constrained noninformative priors in risk assessment," *Reliability
  Engineering & System Safety* 53(1), 37–46, 1996.
* **R. E. Kass and D. Steffey**, "Approximate Bayesian inference in conditionally independent
  hierarchical models (parametric empirical Bayes models)," *Journal of the American
  Statistical Association* 84(407), 717–726, 1989.
* **M. S. Hamada, A. G. Wilson, C. S. Reese and H. F. Martz**, *Bayesian Reliability*, Springer,
  2008, and **H. F. Martz and R. A. Waller**, *Bayesian Reliability Analysis*, Wiley, 1982: the
  textbook treatments.
* **J. G. Ibrahim and M.-H. Chen**, "Power prior distributions for regression models,"
  *Statistical Science* 15(1), 46–60, 2000: the power prior used in Lesson 2.

These are the methods. The project applies them; it does not extend them.

## 3. Hierarchical Bayes for fleets and aviation

* **R. M. Arnaldo Valdés, V. F. Gómez Comendador, L. Pérez Sanz and Á. Rodríguez Sanz**,
  "Prediction of aircraft safety incidents using Bayesian inference and hierarchical
  structures," *Safety Science* 104, 216–230, 2018. Hierarchical Gamma-Poisson and
  Beta-binomial models of monthly incident data across aircraft fleets, so that incident
  rates can differ by fleet type. **The closest work found**: the same statistical structure
  in aviation, applied to safety incidents rather than maintenance alerting, and without the
  small-fleet versus classical alert-level comparison.
* **R. M. Arnaldo Valdés and V. F. Gómez Comendador**, "Hierarchical Bayesian models to estimate
  the number of losses of separation between aircraft in flight," *Applied Sciences* 11(4),
  1600, 2021. The same approach across air-traffic-management units.
* **F. U. Küçüker and B. Yet**, "Reliability prediction for aircraft fleet operators: a Bayesian
  network model that combines supplier estimates, maintenance data and expert judgement,"
  *Journal of the Operational Research Society* 74(10), 2187–2198, 2023. It combines design-life
  estimates, expert judgement and operator data, and is most useful when a fleet is young
  and data are scarce. This is the same motivation as Lesson 2, with a different mechanism (a
  Bayesian network, not conjugate borrowing with a measured cost of bias).
* **C. Lewis-Beck, Q. Tian and W. Q. Meeker**, "Prediction of future failures for heterogeneous
  reliability field data," *Technometrics* 64(1), 125–138, 2021 (online). Hierarchical partial
  pooling of lifetime distributions across subpopulations, so that prediction intervals stay
  stable with few failures, checked by simulation of coverage.
* **L. A. Bull, D. Di Francesco, M. Dhada, O. Steinert, T. Lindgren, A. K. Parlikad, A. B. Duncan
  and M. Girolami**, "Hierarchical Bayesian modelling for knowledge transfer across engineering
  fleets via multitask learning," *Computer-Aided Civil and Infrastructure Engineering*, 2022
  (arXiv:2204.12404). Population-level hierarchical models for a truck fleet and a wind farm.

## 4. Bayesian monitoring of counts

* **K. Bourazas, D. Kiagias and P. Tsiamyrtzis**, "Predictive control charts (PCC): a Bayesian
  approach in online monitoring of short runs," *Journal of Quality Technology* 54(4), 367–391,
  2021 (online). Self-starting Bayesian control charts built on the predictive distribution,
  with power priors for historical data, covering the Poisson case and short runs. **The
  closest work to Lesson 5's alert**, in industrial process monitoring rather than aircraft
  reliability programmes. It is implemented in the R package `bayespm`.

## 5. Data

* **A. Saxena, K. Goebel, D. Simon and N. Eklund**, "Damage propagation modeling for aircraft
  engine run-to-failure simulation," *International Conference on Prognostics and Health
  Management*, 2008; **A. Saxena and K. Goebel**, *Turbofan Engine Degradation Simulation Data
  Set*, NASA Ames Prognostics Data Repository, 2008 (Lesson 8).

## 6. What is, and is not, new here

**Not new:** the Gamma-Poisson model, the Jeffreys and power priors, empirical Bayes with the
Kass-Steffey correction, hierarchical Bayes with MCMC, posterior predictive checks,
simulation-based calibration, Bayesian control charts, renewal theory. All are textbook or
published, and cited above.

**New, narrowly**, as far as the search above could find:

1. A **quantified account of how the classical alert level of AC 120-17B / AMA 571.101/1
   degrades on business-jet-sized fleets**: closed forms for zero-event months and degenerate
   baselines (σ = 0 in 59 % of years on five aircraft), measured false-alarm rates that drift
   from 3 to 11 % with fleet size, and the information limit that bounds any rule on a fleet
   that expects about one extra event a year.
2. A **paired comparison at matched false-alarm rates** of the classical rule against a
   posterior-exceedance alert. It shows the Bayesian rule is a count threshold with an exact
   false-alarm rate, that it dominates on 25 aircraft, and that on 5 its value is control, not
   power.
3. An **exact quadrature oracle** for the hierarchical Gamma-Poisson model that reproduces
   NUREG/CR-6823's WinBUGS results without sampling, used to validate PyMC. It includes a
   documented failure of a Laplace-chosen grid on sparse data.
4. **Failure modes of empirical Bayes on a few sparse operators**: maxima on the zero-spread
   boundary in 40 to 71 % of replicates and 11 to 50 % coverage. Exact hierarchical Bayes on
   the same draws covers 0.90 to 0.99.
5. All of it as **open, tested, reproducible software with a course**, on synthetic fleets
   with known truth and on C-MAPSS lifetimes turned into fleets with a known renewal rate.

**Limits of the search:** web search and the publishers' and Crossref's public records,
November 2026. Paywalled full texts were not read beyond their abstracts. A systematic search
of *Reliability Engineering & System Safety*, *IEEE Transactions on Reliability*, the RAMS
and PSAM proceedings, and military small-fleet reliability reports could still find closer
work. Until it is done, items 1, 2 and 4 should be stated as "not found", not "not done".
