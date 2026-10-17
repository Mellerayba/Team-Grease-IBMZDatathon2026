"""Download all open datasets into notebooks/data/raw/ (resumable: existing files are skipped).

    python scripts/fetch_data.py              # everything
    python scripts/fetch_data.py small        # everything except the large EMS incident pull
    python scripts/fetch_data.py ems          # only EMS incidents
    python scripts/fetch_data.py census       # only Census ACS (needs CENSUS_API_KEY)

Optional env vars:
    SOCRATA_APP_TOKEN  raises NYC Open Data rate limits (free). The team's token is in notebooks/HarryRowlands.ipynb
    CENSUS_API_KEY     required for ACS; free at https://api.census.gov/data/key_signup.html
    EMS_YEARS          comma list, default 2019,2021,2022 (2020 skipped: COVID distorts demand;
                       weather file ends 2022-10-25)
"""
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "notebooks" / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
NYC = "https://data.cityofnewyork.us"
NYS = "https://health.data.ny.gov"
HEADERS = {"X-App-Token": os.environ["SOCRATA_APP_TOKEN"]} if os.environ.get("SOCRATA_APP_TOKEN") else {}

EMS_COLS = [
    "incident_id", "incident_datetime", "initial_call_type", "initial_severity_level_code",
    "final_call_type", "final_severity_level_code", "first_assignment_datetime",
    "valid_dispatch_rspns_time_indc", "dispatch_response_seconds_qy", "first_activation_datetime",
    "first_on_scene_datetime", "valid_incident_rspns_time_indc", "incident_response_seconds_qy",
    "incident_travel_tm_seconds_qy", "first_to_hosp_datetime", "first_hosp_arrival_datetime",
    "incident_close_datetime", "held_indicator", "borough", "incident_dispatch_area", "zipcode",
    "policeprecinct", "citycouncildistrict", "communitydistrict", "special_event_indicator",
    "standby_indicator", "transfer_indicator",
]


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def socrata_csv(base, rid, out, params=None, page=50_000):
    """Page a Socrata resource to a gzipped CSV."""
    if out.exists():
        print(f"skip {out.name}"); return
    params = dict(params or {})
    frames, offset = [], 0
    while True:
        r = get(f"{base}/resource/{rid}.csv", params={**params, "$limit": page, "$offset": offset})
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        if df.empty:
            break
        frames.append(df); offset += len(df)
        print(f"  {out.name}: {offset:,} rows", flush=True)
        if len(df) < page:
            break
    tmp = out.with_name(out.name + ".part")       # write then rename, so an interrupted run never leaves a truncated file
    pd.concat(frames).to_csv(tmp, index=False, compression="gzip" if out.name.endswith(".gz") else None)
    tmp.replace(out)
    print(f"wrote {out.name}")


def fetch_ems():
    """One file per month (small offsets are much faster than deep paging a 30M-row table), then one file per year."""
    years = [int(y) for y in os.environ.get("EMS_YEARS", "2019,2021,2022").split(",")]
    parts = RAW / "ems_parts"; parts.mkdir(exist_ok=True)
    for y in years:
        year_file = RAW / f"ems_incidents_{y}.csv.gz"
        if year_file.exists():
            print(f"skip {year_file.name}"); continue
        month_files = []
        for m in range(1, 13):
            end = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthBegin(1)
            out = parts / f"ems_{y}_{m:02d}.csv.gz"
            socrata_csv(NYC, "76xm-jjuj", out, {
                "$select": ",".join(EMS_COLS),
                "$where": f"incident_datetime >= '{y}-{m:02d}-01T00:00:00' and incident_datetime < '{end:%Y-%m-%d}T00:00:00'",
                "$order": "incident_datetime,incident_id",
            })
            month_files.append(out)
        pd.concat(pd.read_csv(f, dtype=str) for f in month_files).to_csv(year_file, index=False)
        print(f"wrote {year_file.name}")


def fetch_small():
    socrata_csv(NYC, "j34j-vqvt", RAW / "fdny_monthly_response_times.csv", {"$order": "yearmonth"})
    socrata_csv(NYC, "hc8x-tcnd", RAW / "fdny_firehouses.csv")
    socrata_csv(NYS, "vn5v-hh5r", RAW / "nys_hospitals_nyc.csv", {
        "$where": "description like 'Hospital%' AND county in ('New York','Kings','Queens','Bronx','Richmond')"})
    for name, rid in [("modzcta_boundaries", "pri4-ifjk"), ("community_district_boundaries", "5crt-au7u")]:
        out = RAW / f"{name}.geojson"
        if not out.exists():
            out.write_bytes(get(f"{NYC}/resource/{rid}.geojson", params={"$limit": 1000}).content)
            print(f"wrote {out.name}")


def fetch_census():
    out = RAW / "census_acs5_2022_zcta.csv"
    if out.exists():
        print(f"skip {out.name}"); return
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        print("!! CENSUS_API_KEY not set. Get a free key at https://api.census.gov/data/key_signup.html "
              "then rerun: CENSUS_API_KEY=... python scripts/fetch_data.py census"); return
    vars_ = {"B01003_001E": "population", "B19013_001E": "median_household_income",
             "B25044_001E": "occupied_housing_units", "B25044_003E": "owner_no_vehicle",
             "B25044_010E": "renter_no_vehicle", "B17001_001E": "poverty_universe",
             "B17001_002E": "below_poverty", "B01002_001E": "median_age"}
    r = get("https://api.census.gov/data/2022/acs/acs5",
            params={"get": ",".join(vars_), "for": "zip code tabulation area:*", "key": key})
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns={**vars_, "zip code tabulation area": "zcta"})
    df.to_csv(out, index=False); print(f"wrote {out.name}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "small"): fetch_small()
    if what in ("all", "census", "small"): fetch_census()
    if what in ("all", "ems"): fetch_ems()
