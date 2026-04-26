"""More advanced shape generators.

Builds new candidate shapes for Revenue/COGS forecasts that combine
day-of-year (seasonal), day-of-week (weekly), month (monthly), and
a recency-weighted multi-year average.

Each shape is normalized so that it can be used as a multiplicative
baseline: divide by its own annual mean, then scale to our target
annual mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "sales.csv"
SAMPLE_FILE = ROOT / "data" / "sample_submission.csv"


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def _decay_weights(years: pd.Series, ref_year: int, half_life_years: float) -> np.ndarray:
    if half_life_years <= 0:
        return np.ones(len(years))
    age = ref_year - years.to_numpy()
    return 0.5 ** (age / max(half_life_years, 1e-6))


def shape_doy_weighted(
    train: pd.DataFrame,
    target_dates: pd.Series,
    target: str,
    year_window: tuple[int, int],
    half_life_years: float,
    smooth_window: int = 0,
) -> np.ndarray:
    sub = train[(train["Date"].dt.year >= year_window[0]) & (train["Date"].dt.year <= year_window[1])].copy()
    sub["doy"] = sub["Date"].dt.dayofyear
    ref_year = year_window[1]
    sub["weight"] = _decay_weights(sub["Date"].dt.year, ref_year, half_life_years)
    shape = sub.groupby("doy").apply(lambda g: np.average(g[target], weights=g["weight"]))
    if smooth_window > 0:
        shape = shape.rolling(smooth_window, center=True, min_periods=1).mean()
    doys = pd.to_datetime(target_dates).dt.dayofyear
    return doys.map(shape).to_numpy()


def shape_doy_dow_blend(
    train: pd.DataFrame,
    target_dates: pd.Series,
    target: str,
    year_window: tuple[int, int],
    half_life_years: float,
    smooth_window: int,
    dow_weight: float,
) -> np.ndarray:
    sub = train[(train["Date"].dt.year >= year_window[0]) & (train["Date"].dt.year <= year_window[1])].copy()
    sub["doy"] = sub["Date"].dt.dayofyear
    sub["dow"] = sub["Date"].dt.dayofweek
    sub["month"] = sub["Date"].dt.month
    ref_year = year_window[1]
    sub["weight"] = _decay_weights(sub["Date"].dt.year, ref_year, half_life_years)

    doy_shape = sub.groupby("doy").apply(lambda g: np.average(g[target], weights=g["weight"]))
    if smooth_window > 0:
        doy_shape = doy_shape.rolling(smooth_window, center=True, min_periods=1).mean()

    month_dow_shape = sub.groupby(["month", "dow"]).apply(lambda g: np.average(g[target], weights=g["weight"]))
    month_shape = sub.groupby("month").apply(lambda g: np.average(g[target], weights=g["weight"]))

    month_dow_index = {k: float(month_dow_shape.loc[k]) for k in month_dow_shape.index}
    month_index = {int(k): float(month_shape.loc[k]) for k in month_shape.index}

    dates = pd.to_datetime(target_dates)
    doys = dates.dt.dayofyear
    months = dates.dt.month
    dows = dates.dt.dayofweek
    p_doy = doys.map(doy_shape).to_numpy()

    # dow multiplier: ratio of (month, dow) mean to (month) mean
    dow_mult = np.array([month_dow_index[(m, d)] / month_index[m] for m, d in zip(months, dows)])

    # Blend: p_doy * (1 - dow_weight + dow_weight * dow_mult)
    return p_doy * ((1 - dow_weight) + dow_weight * dow_mult)


def shape_lag_multi_year(
    train: pd.DataFrame,
    target_dates: pd.Series,
    target: str,
    lag_years: list[int],
    half_life_years: float,
) -> np.ndarray:
    """For each target date, take value from date - lag for each lag in lag_years, weight and average."""
    idx = train.set_index("Date")[target]
    dates = pd.to_datetime(target_dates)
    out = np.zeros(len(dates))
    for i, d in enumerate(dates):
        vals = []
        weights = []
        for lag in lag_years:
            anchor = d - pd.DateOffset(years=lag)
            if anchor in idx.index:
                vals.append(float(idx.loc[anchor]))
                w = 0.5 ** (lag / max(half_life_years, 1e-6)) if half_life_years > 0 else 1.0
                weights.append(w)
        if vals:
            out[i] = np.average(vals, weights=weights)
        else:
            out[i] = float(idx.tail(365).mean())
    return out


def shape_sample(target_dates: pd.Series, target: str) -> np.ndarray:
    """Use sample_submission as a DoY shape (average 2023+2024 per day-of-year)."""
    sample = load_sample()
    sample["doy"] = sample["Date"].dt.dayofyear
    shape = sample.groupby("doy")[target].mean()
    doys = pd.to_datetime(target_dates).dt.dayofyear
    return doys.map(shape).to_numpy()


def apply_per_year_scale(
    pred: np.ndarray, dates: pd.Series, scales: dict[int, float]
) -> np.ndarray:
    out = pred.copy().astype(float)
    years = pd.to_datetime(dates).dt.year.to_numpy()
    for y, s in scales.items():
        out[years == y] = pred[years == y] * s
    return out


def best_year_scales(pred: np.ndarray, actual: np.ndarray, years: np.ndarray) -> dict[int, float]:
    out = {}
    for y in sorted(set(years)):
        mask = years == y
        if mask.sum() == 0:
            continue
        # MAE-optimal scalar = median(actual/pred)
        ratios = actual[mask] / np.where(pred[mask] == 0, 1.0, pred[mask])
        out[int(y)] = float(np.median(ratios))
    return out


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_holdout_experiments() -> pd.DataFrame:
    train = load_train()
    horizon = 548
    fit = train.iloc[: len(train) - horizon].reset_index(drop=True)
    holdout = train.iloc[len(train) - horizon :].reset_index(drop=True)
    dates = holdout["Date"]
    actual_rev = holdout["Revenue"].to_numpy()
    actual_cog = holdout["COGS"].to_numpy()
    years = dates.dt.year.to_numpy()

    experiments: list[tuple[str, Callable]] = [
        ("doy_19_20", lambda t: shape_doy_weighted(fit, dates, t, (2019, 2020), 0.0, 0)),
        ("doy_19_20_s7", lambda t: shape_doy_weighted(fit, dates, t, (2019, 2020), 0.0, 7)),
        ("doy_18_21", lambda t: shape_doy_weighted(fit, dates, t, (2018, 2021), 0.0, 0)),
        ("doy_18_21_hl2", lambda t: shape_doy_weighted(fit, dates, t, (2018, 2021), 2.0, 0)),
        ("doy_18_21_hl1", lambda t: shape_doy_weighted(fit, dates, t, (2018, 2021), 1.0, 0)),
        ("doy_18_21_hl05_s3", lambda t: shape_doy_weighted(fit, dates, t, (2018, 2021), 0.5, 3)),
        ("doy_15_21_hl2_s7", lambda t: shape_doy_weighted(fit, dates, t, (2015, 2021), 2.0, 7)),
        ("doy_dow_blend_18_21_hl2_dw02_s5", lambda t: shape_doy_dow_blend(fit, dates, t, (2018, 2021), 2.0, 5, 0.2)),
        ("doy_dow_blend_18_21_hl2_dw04_s3", lambda t: shape_doy_dow_blend(fit, dates, t, (2018, 2021), 2.0, 3, 0.4)),
        ("doy_dow_blend_19_21_hl1_dw03_s0", lambda t: shape_doy_dow_blend(fit, dates, t, (2019, 2021), 1.0, 0, 0.3)),
        ("lag_multi_year_hl2", lambda t: shape_lag_multi_year(fit, dates, t, [1, 2, 3], 2.0)),
        ("sample", lambda t: shape_sample(dates, t)),
    ]

    rows = []
    for name, fn in experiments:
        try:
            rev_pred_raw = fn("Revenue")
            cog_pred_raw = fn("COGS")
        except Exception as e:
            print(f"{name}: ERROR {e}")
            continue

        rev_scales = best_year_scales(rev_pred_raw, actual_rev, years)
        cog_scales = best_year_scales(cog_pred_raw, actual_cog, years)
        rev_pred = apply_per_year_scale(rev_pred_raw, dates, rev_scales)
        cog_pred = apply_per_year_scale(cog_pred_raw, dates, cog_scales)

        m_rev = metrics(actual_rev, rev_pred)
        m_cog = metrics(actual_cog, cog_pred)
        rows.append({
            "name": name,
            "rev_mae": m_rev["mae"],
            "cog_mae": m_cog["mae"],
            "rev_rmse": m_rev["rmse"],
            "cog_rmse": m_cog["rmse"],
            "rev_r2": m_rev["r2"],
            "cog_r2": m_cog["r2"],
            "rev_scales": rev_scales,
            "cog_scales": cog_scales,
            "lb_proxy": m_rev["mae"] + m_cog["mae"],
        })
    return pd.DataFrame(rows).sort_values("lb_proxy").reset_index(drop=True)


if __name__ == "__main__":
    df = run_holdout_experiments()
    print(df.to_string(index=False))
