"""Relocation optimizer (dynamic MEXCLP / demand-weighted travel time) via OR-Tools or PuLP.

Constraints: max relocation distance, max moves per unit per shift, min coverage per borough.
"""
from grease.config import SimConfig


def solve_relocation(idle_units, posts, expected_demand, travel, cfg: SimConfig = SimConfig()) -> dict:
    """Return {unit_id: post_id}. Must run in seconds. TODO."""
    raise NotImplementedError
