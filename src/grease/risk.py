"""Risk index (decision D9): a derived, explainable layer on top of the demand forecast. Not a model, so the weights are explicit assumptions.

    need   = w_critical * expected_critical + w_other * expected_other          (critical calls count more)
    raw    = need * (1 + alpha * vulnerability)                                 (vulnerability in [0, 1])
    index  = percentile rank of raw across zones, 0-100                         (for the map)

Vulnerability is the average percentile rank (0-1) of poverty, households without a vehicle and asthma rate. Later it can add the
coverage gap (expected response time minus the 8-minute target) once the simulator exists.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    w_critical: float = 3.0          # ASSUMPTION: a critical call is worth 3 other calls
    w_other: float = 1.0
    alpha: float = 0.5               # ASSUMPTION: the most vulnerable zone's need is scaled up by 50%
    vulnerability_cols: tuple = ("pct_poverty", "pct_no_vehicle", "asthma_rate_per_10k")


def vulnerability(zone_ctx: pd.DataFrame, cfg: RiskConfig = RiskConfig()) -> pd.Series:
    """Per-zone vulnerability in [0, 1]; `zone_ctx` is indexed by zone. Zones missing every column get the median (0.5)."""
    cols = [c for c in cfg.vulnerability_cols if c in zone_ctx.columns]
    if not cols:
        return pd.Series(0.5, index=zone_ctx.index)
    return zone_ctx[cols].rank(pct=True).mean(axis=1).fillna(0.5)


def risk_index(expected_critical: pd.Series, expected_other: pd.Series, vuln: pd.Series, cfg: RiskConfig = RiskConfig()) -> pd.Series:
    """All three inputs are indexed by zone. Returns 0-100 (percentile rank among the zones passed in)."""
    need = cfg.w_critical * expected_critical + cfg.w_other * expected_other
    raw = need * (1 + cfg.alpha * vuln.reindex(need.index).fillna(0.5))
    return raw.rank(pct=True) * 100
