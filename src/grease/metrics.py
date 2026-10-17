"""Evaluation metrics: mean/P90 response, % within target, borough equity gaps, relocation count."""
import pandas as pd


def summarize(results: pd.DataFrame, target_min: float = 8.0) -> dict:
    raise NotImplementedError
