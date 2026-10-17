"""Baselines the model must beat (decision M4). All are computed on the same target as the model: incidents in the next k buckets.

- historical_mean: mean of the target for the same zipcode x weekday x time of day in the training period
- surge_multiplier: citywide recent activity (last 2 h) relative to normal for that weekday and time, clipped; multiply onto historical_mean
- last_week: the same target observed one week earlier (needs no training period)
"""
import numpy as np
import pandas as pd

BUCKETS_PER_WEEK = 336


def _cell(df: pd.DataFrame) -> np.ndarray:
    """Integer id of (zipcode, weekday, half-hour of day)."""
    half_hour = (df["hour"].to_numpy() * 2).round().astype("int64")
    return (df["zipcode"].cat.codes.to_numpy().astype("int64") * 7 + df["dow"].to_numpy().astype("int64")) * 48 + half_hour


def historical_mean(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Per-row prediction for `test`. Unseen cells fall back to the zipcode mean, then the overall mean."""
    n_zip = len(train["zipcode"].cat.categories)
    size = n_zip * 7 * 48
    ok = train[target].notna().to_numpy()
    cell, y = _cell(train)[ok], train[target].to_numpy()[ok].astype("float64")
    total, count = np.bincount(cell, weights=y, minlength=size), np.bincount(cell, minlength=size)
    cell_mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    zip_codes = train["zipcode"].cat.codes.to_numpy()[ok].astype("int64")
    zip_total, zip_count = np.bincount(zip_codes, weights=y, minlength=n_zip), np.bincount(zip_codes, minlength=n_zip)
    zip_mean = np.where(zip_count > 0, zip_total / np.maximum(zip_count, 1), y.mean())

    pred = cell_mean[_cell(test)]
    fallback = zip_mean[test["zipcode"].cat.codes.to_numpy().astype("int64")]
    return np.where(np.isnan(pred), fallback, pred)


def surge_multiplier(train: pd.DataFrame, test: pd.DataFrame, lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """Citywide ratio of the last-2-hours count to its historical mean for that weekday and half-hour (1.0 where unknown)."""
    def keys(index):
        return pd.MultiIndex.from_arrays([index.dayofweek, index.hour + index.minute / 60])
    city_train = train.groupby("bucket", observed=True)["roll_2h"].sum()
    normal = city_train.groupby(keys(city_train.index)).mean()
    city_test = test.groupby("bucket", observed=True)["roll_2h"].sum()
    expected = normal.reindex(keys(city_test.index)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):               # cells with no history give inf / NaN -> clipped / set to 1.0
        ratio = city_test.to_numpy() / expected
    mult = pd.Series(np.clip(ratio, lo, hi), index=city_test.index).fillna(1.0)
    return test["bucket"].map(mult).to_numpy(dtype="float64")


def last_week(panel: pd.DataFrame, target: str) -> pd.Series:
    """The target one week earlier for the same zipcode. Fully observed at issue time because every horizon is shorter than a week."""
    return panel.groupby("zipcode", observed=True)[target].shift(BUCKETS_PER_WEEK)
