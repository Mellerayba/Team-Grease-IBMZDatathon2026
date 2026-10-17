"""Scoring (decisions F13, M6): error overall, on heat days, per borough, on the busiest zips, and calibration of the predicted magnitudes."""
import numpy as np
import pandas as pd


def poisson_deviance(y, mu) -> float:
    y, mu = np.asarray(y, dtype="float64"), np.clip(np.asarray(mu, dtype="float64"), 1e-6, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


def mae(y, mu) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype="float64") - np.asarray(mu, dtype="float64"))))


def score_slices(actual: np.ndarray, preds: dict, slices: dict) -> pd.DataFrame:
    """One row per (slice, model). Every model is scored on the SAME rows (rows where any prediction or the actual is missing are dropped)."""
    actual = np.asarray(actual, dtype="float64")
    valid = ~np.isnan(actual)
    for p in preds.values():
        valid &= ~np.isnan(np.asarray(p, dtype="float64"))
    rows = []
    for sname, smask in slices.items():
        m = valid & np.asarray(smask, dtype=bool)
        if m.sum() == 0:
            continue
        for mname, p in preds.items():
            p = np.asarray(p, dtype="float64")
            rows.append({"slice": sname, "model": mname, "rows": int(m.sum()), "mean_actual": actual[m].mean(), "mean_pred": p[m].mean(),
                         "bias": p[m].mean() / actual[m].mean() if actual[m].mean() > 0 else np.nan,
                         "MAE": mae(actual[m], p[m]), "deviance": poisson_deviance(actual[m], p[m])})
    return pd.DataFrame(rows)


def calibration_table(y, mu, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual mean by prediction decile. ratio near 1 in every row means the magnitudes can be trusted."""
    d = pd.DataFrame({"y": np.asarray(y, dtype="float64"), "mu": np.asarray(mu, dtype="float64")}).dropna()
    d["bin"] = pd.qcut(d["mu"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(rows=("y", "size"), mean_pred=("mu", "mean"), mean_actual=("y", "mean"))
    t["ratio"] = t["mean_pred"] / t["mean_actual"].replace(0, np.nan)
    return t


def busiest_zipcodes(train: pd.DataFrame, n: int = 20, col: str = "y") -> set:
    return set(train.groupby("zipcode", observed=True)[col].sum().nlargest(n).index)
