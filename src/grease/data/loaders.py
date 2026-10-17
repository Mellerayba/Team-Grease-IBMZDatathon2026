"""Load tables produced by notebooks/DarraghC.ipynb (all non-EMS sources). EMS counts come from HarryRowlands.ipynb (ems_raw/)."""
import pandas as pd

from grease.config import DATA_INTERIM, NOTEBOOK_DATA


def load_weather_calendar() -> pd.DataFrame:
    """Weather + calendar features on the 30-minute `bucket` grid (New York local time)."""
    return pd.read_csv(NOTEBOOK_DATA / "weather_calendar_30min.csv.gz", parse_dates=["bucket"])


def load_zone_context() -> pd.DataFrame:
    """One row per zone: borough, location, population, asthma rate, Census fields, hospital access."""
    return pd.read_csv(DATA_INTERIM / "zone_context.csv", dtype={"zone": str})


def load_zip_to_zone() -> pd.DataFrame:
    return pd.read_csv(DATA_INTERIM / "zip_to_zone.csv", dtype=str)


def load_zones() -> pd.DataFrame:
    return pd.read_csv(DATA_INTERIM / "zones.csv", dtype={"zone": str})


def load_posts() -> pd.DataFrame:
    """Candidate staging posts (hospitals + firehouses)."""
    return pd.read_csv(DATA_INTERIM / "posts.csv", dtype={"zone": str})
