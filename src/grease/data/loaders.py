"""Load cleaned datasets produced by notebooks/cleaning/01-07 (see docs/data.md)."""
import pandas as pd

from grease.config import DATA_INTERIM, DATA_PROCESSED


def load_demand(path=DATA_PROCESSED / "demand_zone_30min.csv.gz") -> pd.DataFrame:
    """Zone x 30-min incident counts with weather and calendar features."""
    return pd.read_csv(path, parse_dates=["bucket"], dtype={"zone": str})


def load_incidents(path=DATA_PROCESSED / "incidents_sim.csv.gz") -> pd.DataFrame:
    """One row per incident, for the replay engine and simulator."""
    return pd.read_csv(path, parse_dates=["incident_datetime"], dtype={"zone": str})


def load_zones() -> pd.DataFrame:
    return pd.read_csv(DATA_INTERIM / "zones_modzcta.csv", dtype={"zone": str})


def load_posts() -> pd.DataFrame:
    """Candidate staging posts (hospitals + firehouses)."""
    return pd.read_csv(DATA_INTERIM / "candidate_posts.csv", dtype={"zone": str})
