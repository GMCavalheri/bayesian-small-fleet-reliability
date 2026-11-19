"""Extension: NASA C-MAPSS run-to-failure lifetimes as fleets with a known long-run rate.

Each C-MAPSS sub-dataset (FD001-FD004) is a population of simulated turbofan engines run to
failure (Saxena, Goebel, Simon & Eklund 2008). Treat one engine position on an aircraft as
a renewal process: when an engine fails it is replaced by a new one whose life is drawn from
the same sub-dataset. By the renewal theorem the long-run failure rate is 1/E[life], and a
stationary (equilibrium) renewal process has E[N(T)] = T/E[life] in any window of length T.
That gives each sub-dataset a true rate computed from all its engines, against which
small-fleet estimates can be scored. The lifetimes are wear-out (increasing hazard), so
window counts are less variable than Poisson: a deliberate test of the methods under a
misspecified count model.

The data are downloaded by ``scripts/fetch_cmapss.py`` into ``data/external/`` (checked by
SHA-256, never committed).
"""

from smallfleet.cmapss.loader import SUBSETS, load_lifetimes
from smallfleet.cmapss.renewal import stationary_counts, true_rate

__all__ = ["SUBSETS", "load_lifetimes", "stationary_counts", "true_rate"]
