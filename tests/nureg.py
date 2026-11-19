"""Loader for the NUREG/CR-6823 Example 8.1 fixture (66 plants, 1984 scrams)."""

import csv
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "data" / "nureg6823_example_8_1.csv"


def example_8_1() -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = [line for line in FIXTURE.read_text().splitlines() if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    names = [r["plant"] for r in rows]
    x = np.array([int(r["scrams"]) for r in rows])
    t = np.array([float(r["critical_hours_thousands"]) for r in rows])
    return names, x, t
