"""Load tables produced by notebooks/DarraghC.ipynb (see docs/data.md)."""
import pandas as pd

from grease.config import DATA_INTERIM, DATA_PROCESSED


def load_model_table(path=DATA_PROCESSED / "model_table.csv.gz") -> pd.DataFrame:
    """Zone x 30-min incident counts with weather, calendar and health/zone context (the forecaster's input)."""
    return pd.read_csv(path, parse_dates=["bucket"], dtype={"zone": str})


def load_incidents(path=DATA_PROCESSED / "incidents.csv.gz") -> pd.DataFrame:
    """One row per incident, for the replay engine and simulator."""
    return pd.read_csv(path, parse_dates=["incident_datetime"], dtype={"zone": str, "zipcode": str})


def load_zones() -> pd.DataFrame:
    return pd.read_csv(DATA_INTERIM / "zones.csv", dtype={"zone": str})


def load_posts() -> pd.DataFrame:
    """Candidate staging posts (hospitals + firehouses)."""
    return pd.read_csv(DATA_INTERIM / "posts.csv", dtype={"zone": str})
