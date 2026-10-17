"""ZIP-level predictions -> zone-level (decision M7). Expected counts are additive, so summing is exact: no retraining needed."""
import numpy as np
import pandas as pd


def aggregate_to_zones(frame: pd.DataFrame, zip_to_zone: pd.Series, value_cols: list[str]) -> pd.DataFrame:
    """`frame` has `zipcode` (category or str), `bucket` and the value columns. Returns one row per (zone, bucket), values summed."""
    zips = frame["zipcode"].astype("category")
    zone_of_cat = pd.Series(zips.cat.categories.astype(str)).map(zip_to_zone).to_numpy(dtype=object)
    zone = zone_of_cat[zips.cat.codes.to_numpy()]
    keep = pd.notna(zone)
    out = frame.loc[keep, ["bucket"] + value_cols].copy()
    out["zone"] = zone[keep]
    return out.groupby(["zone", "bucket"], as_index=False)[value_cols].sum()
