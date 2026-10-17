"""Aggregate incidents to (zone, time-bucket) demand counts and estimate service times."""
import pandas as pd


def to_zone_buckets(incidents: pd.DataFrame, bucket_minutes: int = 30) -> pd.DataFrame:
    """Return counts per (zone, bucket_start). TODO."""
    raise NotImplementedError


def estimate_service_times(incidents: pd.DataFrame) -> pd.DataFrame:
    """Empirical on-scene / hospital / return durations from timestamp fields. TODO."""
    raise NotImplementedError
