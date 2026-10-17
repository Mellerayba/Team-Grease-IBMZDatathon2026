"""Baseline: historical mean by zone/weekday/hour x recent-surge multiplier."""
import pandas as pd


class BaselineForecaster:
    def fit(self, buckets: pd.DataFrame) -> "BaselineForecaster":
        raise NotImplementedError

    def predict(self, now: pd.Timestamp, horizon_min: int) -> pd.Series:
        """Expected incidents per zone over the next horizon_min."""
        raise NotImplementedError
