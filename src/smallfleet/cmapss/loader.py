"""Read run-to-failure lifetimes from the C-MAPSS training files.

``train_FD00x.txt`` holds one row per engine per cycle: unit number, cycle, three operating
settings and 21 sensors, space-separated. Every training engine runs until failure, so an
engine's life is its last cycle number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smallfleet.data_generation import IntArray

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def load_lifetimes(directory: Path, subset: str) -> IntArray:
    """Lives (cycles to failure) of every training engine of one sub-dataset, by unit number."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    path = Path(directory) / f"train_{subset}.txt"
    data = np.loadtxt(path, usecols=(0, 1), dtype=np.int64)
    units = np.unique(data[:, 0])
    lives = np.array([data[data[:, 0] == u, 1].max() for u in units], dtype=np.int64)
    if np.any(lives <= 0):
        raise ValueError(f"{path} has a non-positive life")
    return lives
