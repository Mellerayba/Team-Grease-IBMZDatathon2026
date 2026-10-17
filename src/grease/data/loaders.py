"""Load raw datasets (EMS dispatch, weather, census, asthma rates)."""
import pandas as pd

from grease.config import DATA_RAW, NOTEBOOK_DATA


def load_ems_incidents(path=DATA_RAW / "ems_incident_dispatch.csv") -> pd.DataFrame:
    """NYC EMS Incident Dispatch Data. TODO: check geography columns (zip / precinct / CD)."""
    raise NotImplementedError


def load_weather(path=NOTEBOOK_DATA / "NYC_Weather_2016_2022.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    return df.rename(columns=lambda c: c.split(" (")[0])
