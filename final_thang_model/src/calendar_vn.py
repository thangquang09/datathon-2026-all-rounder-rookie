"""Vietnamese e-commerce calendar features.

All dates are public, train-derivable knowledge. No leakage.

Events covered:
- Tet Nguyen Dan (Lunar New Year) — biggest slowdown window; 2012-2024
  dates hard-coded from Vietnamese government calendar.
- 11/11 (Singles Day) and 12/12 e-commerce shopping festivals.
- Black Friday (US-origin but adopted by VN e-commerce,
  last Friday of November).
- Christmas 12/25, Valentine 02/14, Women's Day 08/03, Teachers' Day
  20/11, Hung Kings (lunar 10/3), Mid-autumn (lunar 15/8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Tet Nguyen Dan (Gregorian date of lunar Jan 1) 2012-2026.
# Source: Vietnamese government public calendar.
TET_DATES = {
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
    2026: "2026-02-17",
}

# Mid-autumn festival (lunar Aug 15) — Gregorian:
MID_AUTUMN = {
    2012: "2012-09-30",
    2013: "2013-09-19",
    2014: "2014-09-08",
    2015: "2015-09-27",
    2016: "2016-09-15",
    2017: "2017-10-04",
    2018: "2018-09-24",
    2019: "2019-09-13",
    2020: "2020-10-01",
    2021: "2021-09-21",
    2022: "2022-09-10",
    2023: "2023-09-29",
    2024: "2024-09-17",
}

# Hung Kings Commemoration Day (lunar Mar 10) — Gregorian.
# Cross-checked against Calendarific and Vietnamese news/government-adjacent
# sources.  OfficeHolidays has a known bad 2018 row (Mar 27); the correct
# 2018 date is Apr 25.
HUNG_KINGS = {
    2012: "2012-03-31",
    2013: "2013-04-19",
    2014: "2014-04-09",
    2015: "2015-04-28",
    2016: "2016-04-16",
    2017: "2017-04-06",
    2018: "2018-04-25",
    2019: "2019-04-14",
    2020: "2020-04-02",
    2021: "2021-04-21",
    2022: "2022-04-10",
    2023: "2023-04-29",
    2024: "2024-04-18",
}


def _black_friday(year: int) -> pd.Timestamp:
    """Last Friday of November for a given year."""
    last_day = pd.Timestamp(f"{year}-11-30")
    offset = (last_day.dayofweek - 4) % 7
    return last_day - pd.Timedelta(days=offset)


def _nearest_event_distance(dates: pd.DatetimeIndex, events: list[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
    """For each date, returns (days_to_next_event, days_since_prev_event).
    Clipped to +/- 180 days."""
    event_ts = np.array([e.value for e in sorted(events)])
    date_ts = dates.values.astype("datetime64[ns]").astype(np.int64)
    days_to = np.full(len(dates), 180, dtype=float)
    days_since = np.full(len(dates), 180, dtype=float)
    for i, d in enumerate(date_ts):
        diffs = (event_ts - d) / 86_400_000_000_000  # ns->days
        future = diffs[diffs >= 0]
        past = diffs[diffs <= 0]
        if len(future):
            days_to[i] = min(float(future.min()), 180)
        if len(past):
            days_since[i] = min(float(-past.max()), 180)
    return days_to, days_since


def add_vn_calendar(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Adds Vietnamese calendar / e-commerce event features to df."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    years = sorted(d.dt.year.unique())
    min_year, max_year = min(years), max(years)

    tet_series = [pd.Timestamp(TET_DATES[y]) for y in range(min_year, max_year + 1) if y in TET_DATES]
    mid_autumn_series = [pd.Timestamp(MID_AUTUMN[y]) for y in range(min_year, max_year + 1) if y in MID_AUTUMN]
    hung_series = [pd.Timestamp(HUNG_KINGS[y]) for y in range(min_year, max_year + 1) if y in HUNG_KINGS]
    bf_series = [_black_friday(y) for y in range(min_year, max_year + 1)]

    days_to_tet, days_since_tet = _nearest_event_distance(pd.DatetimeIndex(d), tet_series)
    days_to_ma, _ = _nearest_event_distance(pd.DatetimeIndex(d), mid_autumn_series)
    days_to_hung, days_since_hung = _nearest_event_distance(pd.DatetimeIndex(d), hung_series)
    days_to_bf, days_since_bf = _nearest_event_distance(pd.DatetimeIndex(d), bf_series)

    out["days_to_tet"] = days_to_tet
    out["days_since_tet"] = days_since_tet
    out["is_tet_nguyen_dan"] = (days_since_tet == 0).astype(int)
    out["is_tet_eve_7d"] = ((days_to_tet >= 0) & (days_to_tet <= 7)).astype(int)
    out["is_tet_eve_3d"] = ((days_to_tet >= 0) & (days_to_tet <= 3)).astype(int)
    out["is_tet_after_7d"] = ((days_since_tet >= 0) & (days_since_tet <= 7)).astype(int)
    out["is_tet_after_14d"] = ((days_since_tet >= 0) & (days_since_tet <= 14)).astype(int)
    out["is_tet_window"] = ((days_to_tet <= 3) | (days_since_tet <= 7)).astype(int)

    out["days_to_mid_autumn"] = days_to_ma
    out["is_mid_autumn_day"] = (days_to_ma == 0).astype(int)
    out["is_mid_autumn_eve_7d"] = ((days_to_ma >= 0) & (days_to_ma <= 7)).astype(int)

    out["days_to_hung_kings"] = days_to_hung
    out["days_since_hung_kings"] = days_since_hung
    out["is_hung_kings_day"] = (days_since_hung == 0).astype(int)
    out["is_hung_kings_eve_7d"] = ((days_to_hung >= 0) & (days_to_hung <= 7)).astype(int)
    out["is_hung_kings_after_3d"] = ((days_since_hung >= 0) & (days_since_hung <= 3)).astype(int)
    out["is_hung_kings_window"] = ((days_to_hung <= 3) | (days_since_hung <= 3)).astype(int)

    out["days_to_black_friday"] = days_to_bf
    out["days_since_black_friday"] = days_since_bf
    out["is_black_friday"] = (days_since_bf == 0).astype(int)
    out["is_black_friday_week"] = ((days_to_bf >= 0) & (days_to_bf <= 7)).astype(int)
    out["is_cyber_monday_window"] = ((days_since_bf >= 1) & (days_since_bf <= 3)).astype(int)

    out["is_1111"] = ((d.dt.month == 11) & (d.dt.day == 11)).astype(int)
    out["is_1212"] = ((d.dt.month == 12) & (d.dt.day == 12)).astype(int)
    out["days_to_1111"] = d.apply(
        lambda x: (pd.Timestamp(year=x.year, month=11, day=11) - x).days
        if pd.Timestamp(year=x.year, month=11, day=11) >= x
        else (pd.Timestamp(year=x.year + 1, month=11, day=11) - x).days
    ).clip(0, 180)
    out["days_to_1212"] = d.apply(
        lambda x: (pd.Timestamp(year=x.year, month=12, day=12) - x).days
        if pd.Timestamp(year=x.year, month=12, day=12) >= x
        else (pd.Timestamp(year=x.year + 1, month=12, day=12) - x).days
    ).clip(0, 180)
    out["is_1111_week"] = ((out["days_to_1111"] >= 0) & (out["days_to_1111"] <= 7)).astype(int)
    out["is_1212_week"] = ((out["days_to_1212"] >= 0) & (out["days_to_1212"] <= 7)).astype(int)

    out["is_christmas"] = ((d.dt.month == 12) & (d.dt.day == 25)).astype(int)
    out["is_christmas_eve"] = ((d.dt.month == 12) & (d.dt.day == 24)).astype(int)
    out["is_valentine"] = ((d.dt.month == 2) & (d.dt.day == 14)).astype(int)
    out["is_womens_day_0803"] = ((d.dt.month == 3) & (d.dt.day == 8)).astype(int)
    out["is_womens_day_2010"] = ((d.dt.month == 10) & (d.dt.day == 20)).astype(int)
    out["is_teachers_day_2011"] = ((d.dt.month == 11) & (d.dt.day == 20)).astype(int)
    out["is_reunification_3004"] = ((d.dt.month == 4) & (d.dt.day == 30)).astype(int)
    out["is_labour_0105"] = ((d.dt.month == 5) & (d.dt.day == 1)).astype(int)
    out["is_independence_0209"] = ((d.dt.month == 9) & (d.dt.day == 2)).astype(int)
    out["is_new_year_0101"] = ((d.dt.month == 1) & (d.dt.day == 1)).astype(int)

    out["is_month_payday"] = ((d.dt.day >= 25) | (d.dt.day <= 5)).astype(int)
    out["is_midmonth_payday"] = ((d.dt.day >= 13) & (d.dt.day <= 17)).astype(int)

    dow = d.dt.dayofweek
    day = d.dt.day
    out["is_2nd_saturday"] = ((dow == 5) & (day >= 8) & (day <= 14)).astype(int)
    out["is_long_weekend_mon"] = (
        (dow == 0)
        & (out[[c for c in out.columns if c.startswith("is_") and c.endswith(("_3004", "_0105", "_0209", "_0101", "_2011"))]]
            .shift(1).fillna(0).sum(axis=1) > 0)
    ).astype(int)
    return out


if __name__ == "__main__":
    test = pd.DataFrame({"Date": pd.date_range("2018-01-01", "2024-07-01", freq="D")})
    feat = add_vn_calendar(test)
    print(f"Rows: {len(feat)}, Features added: {feat.shape[1] - 1}")
    for col in ("is_tet_eve_7d", "is_1111", "is_black_friday_week", "is_christmas"):
        n = int(feat[col].sum())
        print(f"  {col}: {n} positive days")
    tet_days = feat.loc[feat["is_tet_eve_3d"] == 1, "Date"].dt.date.tolist()
    print(f"Tet eve (3d) days sample: {tet_days[:6]} ... {tet_days[-3:]}")
