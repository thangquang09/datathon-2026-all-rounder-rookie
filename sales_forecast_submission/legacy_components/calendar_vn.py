"""Vietnamese e-commerce calendar features.

All event dates are deterministically derived from the Gregorian ``Date`` field:

- fixed Gregorian events use month/day rules;
- Black Friday uses the last-Friday-of-November rule;
- Vietnamese lunar events use a solar-to-lunar conversion algorithm for UTC+7.

The module intentionally avoids year-specific holiday lookup tables so feature
generation stays reproducible from contest-provided dates only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


VN_TIMEZONE = 7


def _jd_from_date(day: int, month: int, year: int) -> int:
    """Julian day number from Gregorian date."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jd = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    if jd < 2_299_161:
        jd = day + (153 * m + 2) // 5 + 365 * y + y // 4 - 32_083
    return jd


def _jd_to_date(jd: int) -> tuple[int, int, int]:
    """Gregorian date from Julian day number."""
    if jd > 2_299_160:
        a = jd + 32_044
        b = (4 * a + 3) // 146_097
        c = a - (b * 146_097) // 4
    else:
        b = 0
        c = jd + 32_082
    d = (4 * c + 3) // 1_461
    e = c - (1_461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = b * 100 + d - 4800 + m // 10
    return day, month, year


def _new_moon(k: int) -> float:
    """Astronomical new moon approximation used by Vietnamese lunar calendars."""
    t = k / 1236.85
    t2 = t * t
    t3 = t2 * t
    dr = math.pi / 180
    jd = 2_415_020.75933 + 29.53058868 * k + 0.0001178 * t2 - 0.000000155 * t3
    jd += 0.00033 * math.sin((166.56 + 132.87 * t - 0.009173 * t2) * dr)
    m = 359.2242 + 29.10535608 * k - 0.0000333 * t2 - 0.00000347 * t3
    mpr = 306.0253 + 385.81691806 * k + 0.0107306 * t2 + 0.00001236 * t3
    f = 21.2964 + 390.67050646 * k - 0.0016528 * t2 - 0.00000239 * t3
    correction = (
        (0.1734 - 0.000393 * t) * math.sin(m * dr)
        + 0.0021 * math.sin(2 * dr * m)
        - 0.4068 * math.sin(mpr * dr)
        + 0.0161 * math.sin(2 * dr * mpr)
        - 0.0004 * math.sin(3 * dr * mpr)
        + 0.0104 * math.sin(2 * dr * f)
        - 0.0051 * math.sin((m + mpr) * dr)
        - 0.0074 * math.sin((m - mpr) * dr)
        + 0.0004 * math.sin((2 * f + m) * dr)
        - 0.0004 * math.sin((2 * f - m) * dr)
        - 0.0006 * math.sin((2 * f + mpr) * dr)
        + 0.0010 * math.sin((2 * f - mpr) * dr)
        + 0.0005 * math.sin((2 * mpr + m) * dr)
    )
    if t < -11:
        delta_t = 0.001 + 0.000839 * t + 0.0002261 * t2 - 0.00000845 * t3 - 0.000000081 * t * t3
    else:
        delta_t = -0.000278 + 0.000265 * t + 0.000262 * t2
    return jd + correction - delta_t


def _new_moon_day(k: int, time_zone: int = VN_TIMEZONE) -> int:
    return int(math.floor(_new_moon(k) + 0.5 + time_zone / 24))


def _sun_longitude(jdn: int, time_zone: int = VN_TIMEZONE) -> int:
    """Sun longitude sector, 0..11, at local midnight."""
    t = (jdn - 2_451_545.5 - time_zone / 24) / 36_525
    t2 = t * t
    dr = math.pi / 180
    mean = 357.52910 + 35_999.05030 * t - 0.0001559 * t2 - 0.00000048 * t * t2
    long = 280.46645 + 36_000.76983 * t + 0.0003032 * t2
    dl = (
        (1.914600 - 0.004817 * t - 0.000014 * t2) * math.sin(dr * mean)
        + (0.019993 - 0.000101 * t) * math.sin(2 * dr * mean)
        + 0.000290 * math.sin(3 * dr * mean)
    )
    longitude = (long + dl) * dr
    longitude -= math.pi * 2 * math.floor(longitude / (math.pi * 2))
    return int(math.floor(longitude / math.pi * 6))


def _lunar_month_11(year: int, time_zone: int = VN_TIMEZONE) -> int:
    off = _jd_from_date(31, 12, year) - 2_415_021
    k = int(math.floor(off / 29.530588853))
    nm = _new_moon_day(k, time_zone)
    if _sun_longitude(nm, time_zone) >= 9:
        nm = _new_moon_day(k - 1, time_zone)
    return nm


def _leap_month_offset(month_11: int, time_zone: int = VN_TIMEZONE) -> int:
    k = int(math.floor((month_11 - 2_415_021.076998695) / 29.530588853 + 0.5))
    last = 0
    i = 1
    arc = _sun_longitude(_new_moon_day(k + i, time_zone), time_zone)
    while arc != last and i < 14:
        last = arc
        i += 1
        arc = _sun_longitude(_new_moon_day(k + i, time_zone), time_zone)
    return i - 1


def solar_to_lunar(date: pd.Timestamp | str, time_zone: int = VN_TIMEZONE) -> tuple[int, int, int, int]:
    """Convert Gregorian date to Vietnamese lunar day, month, year, leap flag."""
    ts = pd.Timestamp(date)
    day_number = _jd_from_date(ts.day, ts.month, ts.year)
    k = int(math.floor((day_number - 2_415_021.076998695) / 29.530588853))
    month_start = _new_moon_day(k + 1, time_zone)
    if month_start > day_number:
        month_start = _new_moon_day(k, time_zone)
    a11 = _lunar_month_11(ts.year, time_zone)
    b11 = a11
    if a11 >= month_start:
        lunar_year = ts.year
        a11 = _lunar_month_11(ts.year - 1, time_zone)
    else:
        lunar_year = ts.year + 1
        b11 = _lunar_month_11(ts.year + 1, time_zone)
    lunar_day = day_number - month_start + 1
    diff = int(math.floor((month_start - a11) / 29))
    lunar_leap = 0
    lunar_month = diff + 11
    if b11 - a11 > 365:
        leap_month_diff = _leap_month_offset(a11, time_zone)
        if diff >= leap_month_diff:
            lunar_month = diff + 10
            if diff == leap_month_diff:
                lunar_leap = 1
    if lunar_month > 12:
        lunar_month -= 12
    if lunar_month >= 11 and diff < 4:
        lunar_year -= 1
    return int(lunar_day), int(lunar_month), int(lunar_year), int(lunar_leap)


def _black_friday(year: int) -> pd.Timestamp:
    """Last Friday of November for a given year."""
    last_day = pd.Timestamp(f"{year}-11-30")
    offset = (last_day.dayofweek - 4) % 7
    return last_day - pd.Timedelta(days=offset)


def _gregorian_events_for_year(year: int) -> dict[str, pd.Timestamp]:
    black_friday = _black_friday(year)
    return {
        "new_year": pd.Timestamp(year=year, month=1, day=1),
        "liberation_day": pd.Timestamp(year=year, month=4, day=30),
        "labour_day": pd.Timestamp(year=year, month=5, day=1),
        "national_day": pd.Timestamp(year=year, month=9, day=2),
        "black_friday": black_friday,
        "cyber_monday": black_friday + pd.Timedelta(days=3),
        "valentine": pd.Timestamp(year=year, month=2, day=14),
        "womens_day_mar8": pd.Timestamp(year=year, month=3, day=8),
        "womens_day_oct20": pd.Timestamp(year=year, month=10, day=20),
        "teachers_day": pd.Timestamp(year=year, month=11, day=20),
        "christmas": pd.Timestamp(year=year, month=12, day=25),
        "singles_day_1111": pd.Timestamp(year=year, month=11, day=11),
        "double_1212": pd.Timestamp(year=year, month=12, day=12),
    }


def build_vn_event_calendar(start_year: int, end_year: int) -> pd.DataFrame:
    """Build a machine-readable event table from Gregorian rules only."""
    rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        row: dict[str, object] = {"year": year}
        row.update(_gregorian_events_for_year(year))

        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        lunar = {date: solar_to_lunar(date) for date in dates}
        for date, (day, month, _, leap) in lunar.items():
            if leap:
                continue
            if day == 1 and month == 1:
                row["tet_nguyen_dan"] = date
                row["tet_eve"] = date - pd.Timedelta(days=1)
            elif day == 10 and month == 3:
                row["hung_kings_10_3_lunar"] = date
            elif day == 15 and month == 8:
                row["mid_autumn_15_8_lunar"] = date
        rows.append(row)

    out = pd.DataFrame(rows)
    columns = [
        "year",
        "new_year",
        "tet_eve",
        "tet_nguyen_dan",
        "hung_kings_10_3_lunar",
        "liberation_day",
        "labour_day",
        "national_day",
        "mid_autumn_15_8_lunar",
        "black_friday",
        "cyber_monday",
        "valentine",
        "womens_day_mar8",
        "womens_day_oct20",
        "teachers_day",
        "christmas",
        "singles_day_1111",
        "double_1212",
    ]
    out = out.reindex(columns=columns)
    for col in columns:
        if col != "year":
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    return out


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

    event_table = build_vn_event_calendar(min_year - 1, max_year + 1)
    tet_series = pd.to_datetime(event_table["tet_nguyen_dan"]).dropna().tolist()
    mid_autumn_series = pd.to_datetime(event_table["mid_autumn_15_8_lunar"]).dropna().tolist()
    hung_series = pd.to_datetime(event_table["hung_kings_10_3_lunar"]).dropna().tolist()
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
    print(build_vn_event_calendar(2021, 2024).to_string(index=False))
