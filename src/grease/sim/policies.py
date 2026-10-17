"""Deployment policies compared on the same replayed days."""
from typing import Protocol


class Policy(Protocol):
    def relocate(self, now, idle_units, forecast) -> dict:
        """Return {unit_id: post_id} moves."""


class StaticPolicy:
    """Units stay at fixed home posts."""
    def relocate(self, now, idle_units, forecast) -> dict:
        return {}


class ScheduledPolicy:
    """Relocate by historical hour-of-day averages."""
    def relocate(self, now, idle_units, forecast) -> dict:
        raise NotImplementedError


class ForecastPolicy:
    """Relocate using live forecasts + optimizer."""
    def relocate(self, now, idle_units, forecast) -> dict:
        raise NotImplementedError
