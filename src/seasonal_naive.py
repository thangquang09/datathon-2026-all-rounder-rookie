"""Pure decomposition forecast: DoY-shape × yearly-level.

Compliant: uses ONLY sales.csv. No sample_submission, no test labels.

Method:
  1. For each (year, doy) compute Revenue / mean(year). This gives a
     DoY-relative shape per year.
  2. Average shape across past years (weighted toward recent).
  3. Multiply the shape by an externally chosen yearly mean for 2023, 2024.

This is a strong "seasonal naive with trend" baseline that captures
recurring DoY patterns (weekends, holidays, etc.) without overfitting.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "candidates_v2"


def doy_shape(sales: pd.DataFrame, target: str, weight_recent: float = 1.5) -> pd.Series:
    df = sales.copy()
    df["year"] = df["Date"].dt.year
    df["doy"] = df["Date"].dt.dayofyear

    yearly_mean = df.groupby("year")[target].transform("mean")
    df["ratio"] = df[target] / yearly_mean
    # Weight: years closer to 2022 weigh more
    max_year = df["year"].max()
    df["w"] = weight_recent ** (df["year"] - max_year)  # 2022 = 1, 2021 = 1/1.5, ...
    # Use 2018-2022 only (more representative of recent business)
    df = df[df["year"] >= 2018]

    # Weighted DoY mean ratio
    g = df.groupby("doy").apply(lambda x: np.average(x["ratio"], weights=x["w"]))
    return g  # indexed by doy 1..366


def build(target_levels: dict) -> pd.DataFrame:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    forecast_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
    out = pd.DataFrame({"Date": forecast_dates})
    for col in ("Revenue", "COGS"):
        shape = doy_shape(sales, col)
        doy = out["Date"].dt.dayofyear
        out[col] = doy.map(shape).astype(float)
        # Renormalise per-year so mean exactly hits the target
        out["_y"] = out["Date"].dt.year
        for y, want in target_levels[col].items():
            mask = out["_y"] == y
            cur = out.loc[mask, col].mean()
            if cur > 0:
                out.loc[mask, col] *= want / cur
    out = out.drop(columns=["_y"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    levels = {
        "Revenue": {2023: 4_045_000, 2024: 4_865_000},
        "COGS":    {2023: 3_745_000, 2024: 4_265_000},
    }
    sub = build(levels)
    sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
    sub["Revenue"] = sub["Revenue"].round(2)
    sub["COGS"] = sub["COGS"].round(2)
    sub.to_csv(OUT / "seasonal_naive.csv", index=False)
    print("Saved seasonal_naive.csv")
    s = sub.copy(); s["Date"] = pd.to_datetime(s["Date"]); s["year"] = s["Date"].dt.year
    print(s.groupby("year")[["Revenue","COGS"]].mean().round(0))


if __name__ == "__main__":
    main()
