"""LightGBM Poisson forecaster: lags, time features, weather, holidays."""
import pandas as pd


def build_features(buckets: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


class LGBMForecaster:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LGBMForecaster":
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
