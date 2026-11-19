# `cmapss`: NASA C-MAPSS lifetimes as fleets

**Lesson:** [8 · A dataset with real ground truth](lessons/08-a-dataset-with-real-ground-truth.md) ·
**Code:** [`src/smallfleet/cmapss/`](../src/smallfleet/cmapss/),
[`scripts/fetch_cmapss.py`](../scripts/fetch_cmapss.py)

## Model

Life = last cycle of a training engine. Each engine position is a stationary renewal process
with lives resampled from a subset: the first life is length-biased with a uniform residual,
so $E[N(T)] = T/E[\text{life}]$ exactly. True rate = 1000 / mean life per 1000 cycles.

## Data handling

`scripts/fetch_cmapss.py` downloads the NASA Prognostics Center archive (12.4 MB), checks the
archive and the four training files against recorded SHA-256 values, and writes them to
`data/external/cmapss/`, which git ignores. `validation/cmapss.py` skips with a message when
the data are absent (as in CI).

## API

| function | purpose |
|---|---|
| `load_lifetimes(directory, subset)`, `SUBSETS` | lives per training engine |
| `true_rate(lives)`, `stationary_counts(rng, lives, positions, window)` | renewal fleet |

## Verification

A synthetic fixture in the C-MAPSS layout; stationary mean $T/\mu$ by Monte Carlo;
under-dispersion of wear-out counts ([`tests/test_cmapss.py`](../tests/test_cmapss.py)).
