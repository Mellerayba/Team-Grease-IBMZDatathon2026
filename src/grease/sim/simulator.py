"""SimPy discrete-event simulator: dispatch nearest idle unit; model travel, on-scene, hospital, return."""
from grease.config import SimConfig


def run_simulation(incidents, policy, travel, service_times, cfg: SimConfig = SimConfig()):
    """Return per-incident response times and relocation log. TODO."""
    raise NotImplementedError
