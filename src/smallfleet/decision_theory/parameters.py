"""The decision parameters table."""

from __future__ import annotations

import csv
from importlib import resources


def load_decision_parameters() -> dict[str, float]:
    text = resources.files("smallfleet.data").joinpath("decision_parameters.csv").read_text()
    return {r["parameter"]: float(r["value"]) for r in csv.DictReader(text.splitlines())}
