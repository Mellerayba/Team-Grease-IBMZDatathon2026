"""Zone <-> staging-post travel times from the road network (OSMnx/OSRM) with time-of-day multipliers."""
import numpy as np


def build_travel_matrix(zone_centroids, posts) -> np.ndarray:
    """Return (n_zones, n_posts) free-flow minutes. TODO."""
    raise NotImplementedError


def apply_traffic(matrix: np.ndarray, hour: int) -> np.ndarray:
    raise NotImplementedError
