"""Central configuration and assumptions. State every assumption here so the pitch can cite it."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
NOTEBOOK_DATA = ROOT / "notebooks" / "data"


@dataclass(frozen=True)
class SimConfig:
    bucket_minutes: int = 30            # demand aggregation window
    forecast_horizon_min: int = 120     # 30-180 min look-ahead
    relocation_interval_min: int = 30   # optimizer cadence (15-30)
    fleet_size: int = 100               # ASSUMPTION: no unit-location data; sweep a few sizes
    target_response_min: float = 8.0
    max_relocation_min: float = 15.0    # max travel time per move
    max_moves_per_unit_per_shift: int = 3
    min_units_per_borough: int = 1      # equity floor
    seed: int = 42
