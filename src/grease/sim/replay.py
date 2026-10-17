"""Replay engine: yields historical incidents in timestamp order as if live.

Swap this for a real feed to go live (keep the same iterator interface).
"""
from typing import Iterator

import pandas as pd


def replay(incidents: pd.DataFrame) -> Iterator[dict]:
    for row in incidents.sort_values("timestamp").to_dict("records"):
        yield row
