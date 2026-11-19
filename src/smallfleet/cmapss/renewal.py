"""Stationary renewal counts in a window, from an empirical life distribution."""

from __future__ import annotations

import numpy as np

from smallfleet.data_generation import IntArray


def true_rate(lives: IntArray) -> float:
    """Long-run failures per 1000 cycles of one engine position: 1000 / mean life."""
    return 1000.0 / float(np.mean(lives))


def stationary_counts(
    rng: np.random.Generator, lives: IntArray, positions: int, window: int
) -> int:
    """Failures in ``window`` cycles over ``positions`` engine positions, each a stationary
    renewal process with lives resampled from ``lives``.

    The first life in the window is the equilibrium residual life: an engine drawn with
    probability proportional to its life (length-biased), observed at a uniform point of it.
    That makes the process stationary from the first cycle, so E[count] = positions ·
    window / mean(lives) exactly.
    """
    if positions < 1 or window < 1:
        raise ValueError("need at least one position and a positive window")
    weights = lives / lives.sum()
    total = 0
    for _ in range(positions):
        first = int(rng.choice(lives, p=weights))
        clock = rng.uniform(0.0, first)  # time to the first failure in the window
        while clock <= window:
            total += 1
            clock += int(rng.choice(lives))
    return total
