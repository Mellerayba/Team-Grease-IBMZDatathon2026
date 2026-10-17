import pandas as pd

from grease.data.features import add_weather_calendar


def _wx():
    idx = pd.date_range("2022-07-01", periods=96, freq="30min", name="bucket")
    return pd.DataFrame({"temp_c": range(96), "is_holiday": [0] * 48 + [1] * 48,
                         "is_day_before_holiday": [1] * 48 + [0] * 48, "is_day_after_holiday": 0,
                         "eval_is_heat_day": [0] * 48 + [1] * 48}, index=idx)


def test_weather_matches_bucket():
    out = add_weather_calendar(pd.DataFrame({"bucket": pd.to_datetime(["2022-07-01 05:00", "2022-07-01 05:30"])}), wx=_wx())
    assert list(out.temp_c) == [10, 11]


def test_lag_never_reads_future_and_eval_flags_opt_in():
    df = pd.DataFrame({"bucket": pd.to_datetime(["2022-07-02 03:00"])})
    out = add_weather_calendar(df, lag_buckets=4, wx=_wx())
    assert out.temp_c.iloc[0] == 48 + 6 - 4 and out.is_holiday.iloc[0] == 1     # weather 2h earlier, calendar of the day itself
    assert "eval_is_heat_day" not in out.columns
    assert add_weather_calendar(df, include_eval_flags=True, wx=_wx()).eval_is_heat_day.iloc[0] == 1
