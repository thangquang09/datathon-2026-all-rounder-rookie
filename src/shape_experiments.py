"""Compare several shape-generation strategies on local holdout.

Shapes under test:
1. doy_{year_window} — Day-of-year mean across selected years.
2. sample_submission — The provided sample_submission shape itself.
3. dow_month — Day-of-week x Month mean.
4. lag_365 — Exact last-year value.
5. blend of doy + dow_month.
6. weekly_year_blend: within-year weekly average + yearly trend blend.

Metric: MAE / RMSE / R^2 on the last 548 days of train after optimal
per-year scaling. The local optimal scale is found by 1D grid search over
each year.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "sales.csv"
SAMPLE_FILE = ROOT / "data" / "sample_submission.csv"


@dataclass
class ShapeResult:
    name: str
    rev_mae: float
    cog_mae: float
    rev_rmse: float
    cog_rmse: float
    rev_r2: float
    cog_r2: float
    rev_scale_early: float
    rev_scale_late: float
    cog_scale_early: float
    cog_scale_late: float
    lb_proxy: float


def doy_shape(fit: pd.DataFrame, target: str, year_window: tuple[int, int]) -> pd.Series:
    sub = fit[(fit["Date"].dt.year >= year_window[0]) & (fit["Date"].dt.year <= year_window[1])]
    return sub.groupby(sub["Date"].dt.dayofyear)[target].mean()


def dow_month_shape(fit: pd.DataFrame, target: str, year_window: tuple[int, int]) -> pd.DataFrame:
    sub = fit[(fit["Date"].dt.year >= year_window[0]) & (fit["Date"].dt.year <= year_window[1])].copy()
    sub["dow"] = sub["Date"].dt.dayofweek
    sub["month"] = sub["Date"].dt.month
    return sub.groupby(["month", "dow"])[target].mean()


def lag_365_series(fit: pd.DataFrame, target_dates: pd.Series, target: str) -> np.ndarray:
    # For each date in target_dates, return fit[fit.Date == t - 365 days] target value.
    fit_indexed = fit.set_index("Date")[target]
    out = []
    for d in pd.to_datetime(target_dates):
        anchor = d - pd.Timedelta(days=365)
        if anchor in fit_indexed.index:
            out.append(float(fit_indexed.loc[anchor]))
        else:
            out.append(float(fit_indexed.tail(365).mean()))
    return np.array(out)


def build_shape_preds(fit: pd.DataFrame, holdout: pd.DataFrame, target: str, shape_name: str) -> np.ndarray:
    dates = holdout["Date"]
    if shape_name == "doy_19_20":
        s = doy_shape(fit, target, (2019, 2020))
        return dates.dt.dayofyear.map(s).to_numpy()
    elif shape_name == "doy_17_20":
        s = doy_shape(fit, target, (2017, 2020))
        return dates.dt.dayofyear.map(s).to_numpy()
    elif shape_name == "doy_19_20_smooth7":
        s = doy_shape(fit, target, (2019, 2020)).rolling(7, center=True, min_periods=1).mean()
        return dates.dt.dayofyear.map(s).to_numpy()
    elif shape_name == "doy_19_20_smooth14":
        s = doy_shape(fit, target, (2019, 2020)).rolling(14, center=True, min_periods=1).mean()
        return dates.dt.dayofyear.map(s).to_numpy()
    elif shape_name == "dow_month":
        s = dow_month_shape(fit, target, (2019, 2020))
        keys = list(zip(dates.dt.month, dates.dt.dayofweek))
        return np.array([s.loc[k] if k in s.index else s.mean() for k in keys])
    elif shape_name == "lag_365":
        return lag_365_series(fit, dates, target)
    elif shape_name == "sample":
        sample = pd.read_csv(SAMPLE_FILE, parse_dates=["Date"])
        # Sample covers 2023-2024, holdout covers 2021-07..2022-12. Use doy match with year+2 offset.
        # Use sample's doy means as shape.
        sample["doy"] = sample["Date"].dt.dayofyear
        s = sample.groupby("doy")[target].mean()
        return dates.dt.dayofyear.map(s).to_numpy()
    elif shape_name == "doy_20_21":
        # Include 2021 H1 partial year
        sub = fit[(fit["Date"].dt.year >= 2020) & (fit["Date"].dt.year <= 2021)]
        s = sub.groupby(sub["Date"].dt.dayofyear)[target].mean()
        return dates.dt.dayofyear.map(s).to_numpy()
    elif shape_name == "doy_blend_dowmonth":
        s1 = doy_shape(fit, target, (2019, 2020))
        p1 = dates.dt.dayofyear.map(s1).to_numpy()
        s2 = dow_month_shape(fit, target, (2019, 2020))
        keys = list(zip(dates.dt.month, dates.dt.dayofweek))
        p2 = np.array([s2.loc[k] if k in s2.index else s2.mean() for k in keys])
        # Normalize both to same annual mean
        return 0.6 * p1 + 0.4 * p2
    else:
        raise ValueError(f"unknown shape_name={shape_name}")


def best_per_year_scale(pred: np.ndarray, actual: np.ndarray, years: np.ndarray) -> tuple[float, float]:
    """Find optimal scalar per year that minimizes MAE."""
    unique_years = sorted(set(years))
    scales = {}
    for y in unique_years:
        mask = years == y
        if mask.sum() == 0:
            scales[y] = 1.0
            continue
        # Minimize MAE -> median of actual/pred
        ratios = actual[mask] / np.where(pred[mask] == 0, 1.0, pred[mask])
        scales[y] = float(np.median(ratios))
    if len(unique_years) == 2:
        return scales[unique_years[0]], scales[unique_years[1]]
    if len(unique_years) == 1:
        return scales[unique_years[0]], scales[unique_years[0]]
    return list(scales.values())[0], list(scales.values())[-1]


def evaluate_shape(
    fit: pd.DataFrame,
    holdout: pd.DataFrame,
    shape_name: str,
) -> ShapeResult:
    rev_pred = build_shape_preds(fit, holdout, "Revenue", shape_name)
    cog_pred = build_shape_preds(fit, holdout, "COGS", shape_name)
    years = holdout["Date"].dt.year.to_numpy()
    actual_rev = holdout["Revenue"].to_numpy()
    actual_cog = holdout["COGS"].to_numpy()

    rev_s_early, rev_s_late = best_per_year_scale(rev_pred, actual_rev, years)
    cog_s_early, cog_s_late = best_per_year_scale(cog_pred, actual_cog, years)

    rev_scaled = rev_pred.copy()
    cog_scaled = cog_pred.copy()
    first_year = min(years)
    rev_scaled[years == first_year] *= rev_s_early
    rev_scaled[years != first_year] *= rev_s_late
    cog_scaled[years == first_year] *= cog_s_early
    cog_scaled[years != first_year] *= cog_s_late

    def metrics(actual, pred):
        err = actual - pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return mae, rmse, r2

    rev_mae, rev_rmse, rev_r2 = metrics(actual_rev, rev_scaled)
    cog_mae, cog_rmse, cog_r2 = metrics(actual_cog, cog_scaled)

    return ShapeResult(
        name=shape_name,
        rev_mae=rev_mae,
        cog_mae=cog_mae,
        rev_rmse=rev_rmse,
        cog_rmse=cog_rmse,
        rev_r2=rev_r2,
        cog_r2=cog_r2,
        rev_scale_early=rev_s_early,
        rev_scale_late=rev_s_late,
        cog_scale_early=cog_s_early,
        cog_scale_late=cog_s_late,
        lb_proxy=rev_mae + cog_mae,
    )


def main() -> None:
    horizon = 548
    train = pd.read_csv(TRAIN_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    fit = train.iloc[: len(train) - horizon].reset_index(drop=True)
    holdout = train.iloc[len(train) - horizon :].reset_index(drop=True)

    print(f"Fit: {fit['Date'].min().date()} -> {fit['Date'].max().date()} ({len(fit)} days)")
    print(f"Holdout: {holdout['Date'].min().date()} -> {holdout['Date'].max().date()} ({len(holdout)} days)")

    shape_names = [
        "doy_19_20",
        "doy_17_20",
        "doy_19_20_smooth7",
        "doy_19_20_smooth14",
        "dow_month",
        "lag_365",
        "sample",
        "doy_20_21",
        "doy_blend_dowmonth",
    ]

    rows = []
    for name in shape_names:
        try:
            r = evaluate_shape(fit, holdout, name)
            rows.append(r)
        except Exception as e:
            print(f"{name}: ERROR {e}")

    df = pd.DataFrame([vars(r) for r in rows]).sort_values("lb_proxy")
    print("\nRanking by local MAE-sum proxy:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
