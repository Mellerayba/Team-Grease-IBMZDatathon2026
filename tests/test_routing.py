import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))   # routing_policy.py and ems_replay.py live next to the notebooks
import ems_replay as R                                                         # noqa: E402
from routing_policy import LiveState, PolicyConfig, RoutingPolicy, Unit       # noqa: E402

T0 = pd.Timestamp("2022-07-19 00:00")


def _zones():
    """Three ZIPs on a north-south line, about 3.3 km (9 min at the placeholder speed) apart."""
    return pd.DataFrame({"lat": [40.70, 40.73, 40.76], "lon": [-74.0] * 3, "borough": ["BRONX", "BROOKLYN", "QUEENS"]},
                        index=pd.Index(["10001", "10002", "10003"], name="zipcode"))


def _fc(a, b, c):
    return pd.DataFrame({"pred_critical_1h": [a, b, c], "pred_other_1h": [a, b, c]}, index=_zones().index)


def test_stale_forecast_holds_positions():
    pol = RoutingPolicy(_zones())
    st = LiveState(T0, [Unit("U0", "10001")], forecast_age_min=90)
    dec = pol.reposition(_fc(0, 0, 50), st)
    assert dec.moves == [] and any("holding" in n.lower() for n in dec.notes)


def test_idle_unit_moves_toward_demand_and_stays_put_when_a_move_is_not_worth_it():
    pol = RoutingPolicy(_zones(), PolicyConfig(min_units_per_borough=0))
    dec = pol.reposition(_fc(0, 40, 0), LiveState(T0, [Unit("U0", "10001")]))
    assert [m.to_zone for m in dec.moves] == ["10002"] and dec.moves[0].gain > 0
    # equal demand everywhere and the unit already sits in the middle (where it covers all three): nothing to do
    assert pol.reposition(_fc(10, 10, 10), LiveState(T0, [Unit("U0", "10002")])).moves == []
    # a real improvement exists, but the minimum-gain threshold says it is not worth the disruption
    strict = RoutingPolicy(_zones(), PolicyConfig(min_units_per_borough=0, min_gain_frac=0.99))
    dec = strict.reposition(_fc(0, 40, 0), LiveState(T0, [Unit("U0", "10001")]))
    assert dec.moves == [] and any("minimum-gain" in n for n in dec.notes)


def test_equity_floor_gives_every_borough_a_unit():
    pol = RoutingPolicy(_zones())
    layout = pol.optimal_layout(_fc(100, 1, 1), 3, LiveState(T0, []))
    assert set(_zones().loc[layout, "borough"]) == {"BRONX", "BROOKLYN", "QUEENS"}


def test_critical_dispatch_is_the_fastest_unit():
    pol = RoutingPolicy(_zones())
    st = LiveState(T0, [Unit("far", "10003"), Unit("near", "10001"), Unit("mid", "10002")])
    assert pol.dispatch("10001", "critical", st).unit_id == "near"
    assert pol.dispatch("10001", "critical", LiveState(T0, [Unit("x", "10001", status="busy")])) is None   # nobody idle


def _replay(dynamic, forecast_demand, label=None):
    zips = _zones()
    start, end = T0, T0 + pd.Timedelta("1h")
    inc = pd.DataFrame({"time": [T0 + pd.Timedelta("3min")], "zipcode": ["10001"], "acuity": ["critical"]})
    fc = {start - pd.Timedelta("30min"): _fc(*forecast_demand), start: _fc(*forecast_demand)}
    return R.run_replay(zips, fc, inc, _fc(100, 0, 0), None, dynamic=dynamic, n_units=1, start=start, end=end, label=label)


def test_a_repositioning_unit_stays_dispatchable_while_it_drives():
    r = _replay(True, (0, 40, 0))
    assert len(r["moves"]) == 1 and r["moves"].iloc[0].to_zip == "10002"      # the unit did start repositioning at minute 0...
    assert r["unserved"] == 0 and r["responses"].wait.iloc[0] == 0            # ...yet it took the call at minute 3 instead of leaving it queued
    assert r["responses"].eta.iloc[0] < 3                                      # from where it still was (about 2 minutes), not from the destination


def test_static_policy_ignores_the_forecast_and_replays_are_deterministic():
    a, b = _replay(False, (0, 40, 0)), _replay(False, (0, 0, 40))
    assert a["responses"].equals(b["responses"]) and len(a["moves"]) == 0
    assert _replay(True, (0, 40, 0))["responses"].equals(_replay(True, (0, 40, 0))["responses"])


def test_summary_reports_labels_and_dropped_incidents():
    r1, r2 = _replay(False, (0, 40, 0), label="static"), _replay(True, (0, 40, 0), label="dynamic: forecast")
    overall, _ = R.summarize([r1, r2], _zones())
    assert list(overall.index) == ["static", "dynamic: forecast"] and "dropped" in overall and (overall.dropped == 0).all()
