"""Attach weather and calendar features to any table with a time column.

Single source: the 30-minute table written by notebooks/DarraghC.ipynb (`notebooks/data/weather_calendar_30min.csv.gz`),
already in New York local time. HarryRowlands.ipynb joins the same table onto the EMS panel; use this module for any other table
(forecasts, simulator clocks).

Leakage rule: the time column is the moment the forecast is *issued*. `lag_buckets` shifts the weather further back by whole
30-minute buckets (a model predicting h buckets ahead should pass h). Columns starting with `eval_` use whole-day weather,
which is unknown when a forecast is made: they are only for slicing results (heat days) and are returned only on request.
"""
import pandas as pd

from grease.config import NOTEBOOK_DATA

WEATHER_FILE = NOTEBOOK_DATA / "weather_calendar_30min.csv.gz"
CALENDAR_COLS = ["is_holiday", "is_day_before_holiday", "is_day_after_holiday"]


def load_weather_calendar(path=WEATHER_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["bucket"]).set_index("bucket")


def add_weather_calendar(df: pd.DataFrame, time_col: str = "bucket", lag_buckets: int = 0,
                         include_eval_flags: bool = False, wx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return df with weather + calendar columns added. Rows with no weather get NaN (LightGBM handles it)."""
    wx = load_weather_calendar() if wx is None else wx
    eval_cols = [c for c in wx.columns if c.startswith("eval_")]
    model_cols = [c for c in wx.columns if c not in eval_cols]
    key = (df[time_col] - pd.Timedelta(minutes=30 * lag_buckets)).dt.floor("30min")
    out = df.copy()
    weather = wx[[c for c in model_cols if c not in CALENDAR_COLS]].reindex(key.values)
    calendar = wx[CALENDAR_COLS + (eval_cols if include_eval_flags else [])].reindex(df[time_col].dt.floor("30min").values)
    for part in (weather, calendar):
        part.index = out.index
        out = out.join(part)
    return out
