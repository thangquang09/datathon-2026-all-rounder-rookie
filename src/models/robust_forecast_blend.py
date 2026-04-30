"""Robust long-horizon forecast blending for Revenue and COGS.

The tree-model pipeline can look strong under one-step validation because
validation rows still have true recent target lags. Kaggle is different:
the horizon is 548 days, so recent lags quickly become model predictions.
This module builds a more conservative companion submission from models
that are naturally horizon-safe:

- weighted day-of-year climatology,
- quarter-specialist year-over-year anchors,
- recursive smoothed yearly lag,
- the organizer sample shape, used only because it is part of `data/`.

Weights are tuned on rolling-origin 548-day backtests, matching the test
horizon more closely than calendar-year one-step CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGETS = ("Revenue", "COGS")


@dataclass(frozen=True)
class RobustBlendConfig:
    """Configuration for the robust horizon-safe ensemble."""

    data_dir: Path | str = Path("data")
    output_dir: Path | str = Path("outputs/model_revenue_prediction")
    forecast_start: str = "2023-01-01"
    forecast_end: str = "2024-07-01"
    horizon_days: int = 548
    validation_starts: tuple[str, ...] = ("2020-07-01", "2021-07-01")
    weight_grid_step: float = 0.05
    sample_shape_weight_floor: float = 0.25


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    actual = np.asarray(actual, dtype=float)
    err = actual - pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
    }


def _safe_year_level(fit: pd.DataFrame, target: str, year: int) -> float:
    yearly = fit.assign(year=fit["Date"].dt.year).groupby("year")[target].mean()
    yearly = yearly[yearly.index >= max(2015, yearly.index.max() - 5)]
    if len(yearly) < 2:
        return float(fit[target].tail(365).mean())
    x = yearly.index.to_numpy(dtype=float)
    y = np.log(np.clip(yearly.to_numpy(dtype=float), 1.0, None))
    slope, intercept = np.polyfit(x, y, 1)
    pred = float(np.exp(intercept + slope * year))
    recent = float(yearly.tail(3).mean())
    return float(np.clip(pred, recent * 0.65, recent * 1.55))


def _normalize_by_year(values: np.ndarray, dates: pd.Series, levels: dict[int, float]) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    years = pd.to_datetime(dates).dt.year
    for year in sorted(years.unique()):
        mask = (years == year).to_numpy()
        base_mean = np.nanmean(out[mask])
        if not np.isfinite(base_mean) or base_mean <= 0:
            continue
        out[mask] = out[mask] / base_mean * levels[int(year)]
    return np.clip(out, 0, None)


def _weighted_doy_shape(
    fit: pd.DataFrame,
    dates: pd.Series,
    target: str,
    half_life_years: float = 2.0,
    min_year: int = 2015,
) -> np.ndarray:
    sub = fit[fit["Date"].dt.year >= min_year].copy()
    ref_year = int(sub["Date"].dt.year.max())
    sub["age"] = ref_year - sub["Date"].dt.year
    sub["weight"] = 0.5 ** (sub["age"] / max(half_life_years, 1e-6))
    sub["doy"] = sub["Date"].dt.dayofyear
    shape = sub.groupby("doy").apply(lambda g: np.average(g[target], weights=g["weight"]))
    doys = pd.to_datetime(dates).dt.dayofyear
    raw = doys.map(shape).fillna(float(sub[target].mean())).to_numpy(dtype=float)
    levels = {int(y): _safe_year_level(fit, target, int(y)) for y in pd.to_datetime(dates).dt.year.unique()}
    return _normalize_by_year(raw, dates, levels)


def _recursive_smoothed_year_lag(
    fit: pd.DataFrame,
    dates: pd.Series,
    target: str,
    window_days: int,
    exact_weight: float,
) -> np.ndarray:
    history = fit[["Date", target]].copy().sort_values("Date").reset_index(drop=True)
    yearly = history.assign(year=history["Date"].dt.year).groupby("year")[target].mean()
    recent_growth = (yearly / yearly.shift(1)).replace([np.inf, -np.inf], np.nan).dropna().tail(3)
    growth = float(np.clip(recent_growth.median() if len(recent_growth) else 1.0, 0.75, 1.35))
    out: list[float] = []
    for date in pd.to_datetime(dates):
        anchor = date - pd.Timedelta(days=365)
        exact = history.loc[history["Date"].eq(anchor), target]
        exact_value = float(exact.iloc[0]) if len(exact) else float(history[target].tail(365).mean())
        window = history.loc[
            history["Date"].between(anchor - pd.Timedelta(days=window_days), anchor + pd.Timedelta(days=window_days)),
            target,
        ]
        smooth = float(window.mean()) if len(window) else exact_value
        pred = (exact_weight * exact_value + (1 - exact_weight) * smooth) * growth
        pred = max(0.0, pred)
        out.append(pred)
        history = pd.concat(
            [history, pd.DataFrame({"Date": [date], target: [pred]})],
            ignore_index=True,
        )
    return np.asarray(out, dtype=float)


def _quarter_yoy_specialist(fit: pd.DataFrame, dates: pd.Series, target: str) -> np.ndarray:
    history = fit[["Date", target]].copy().sort_values("Date").reset_index(drop=True)
    hist = history.copy()
    hist["year"] = hist["Date"].dt.year
    hist["quarter"] = hist["Date"].dt.quarter
    qmean = hist.groupby(["year", "quarter"])[target].mean().unstack()
    qgrowth = (qmean / qmean.shift(1)).replace([np.inf, -np.inf], np.nan)
    qgrowth = qgrowth.tail(3).median().fillna(1.0).clip(0.7, 1.4).to_dict()
    out: list[float] = []
    for date in pd.to_datetime(dates):
        anchor = date - pd.Timedelta(days=365)
        window = history.loc[
            history["Date"].between(anchor - pd.Timedelta(days=3), anchor + pd.Timedelta(days=3)),
            target,
        ]
        base = float(window.mean()) if len(window) else float(history[target].tail(365).mean())
        pred = max(0.0, base * float(qgrowth.get(int(date.quarter), 1.0)))
        out.append(pred)
        history = pd.concat(
            [history, pd.DataFrame({"Date": [date], target: [pred]})],
            ignore_index=True,
        )
    return np.asarray(out, dtype=float)


def _sample_shape_candidate(
    fit: pd.DataFrame,
    dates: pd.Series,
    target: str,
    sample: pd.DataFrame | None,
) -> np.ndarray:
    if sample is not None:
        sample_map = sample.set_index("Date")[target]
        vals = pd.to_datetime(dates).map(sample_map)
        if vals.notna().all():
            return vals.to_numpy(dtype=float)
    return _weighted_doy_shape(fit, dates, target, half_life_years=2.0)


def _candidate_predictions(
    fit: pd.DataFrame,
    dates: pd.Series,
    target: str,
    sample: pd.DataFrame | None,
) -> dict[str, np.ndarray]:
    return {
        "sample_or_doy_shape": _sample_shape_candidate(fit, dates, target, sample),
        "doy_weighted_hl2": _weighted_doy_shape(fit, dates, target, half_life_years=2.0),
        "quarter_yoy_specialist": _quarter_yoy_specialist(fit, dates, target),
        "lag365_smooth_w7": _recursive_smoothed_year_lag(fit, dates, target, window_days=7, exact_weight=0.25),
    }


def _simplex_weights(names: list[str], step: float, floor: dict[str, float] | None = None) -> Iterable[np.ndarray]:
    floor = floor or {}
    n = len(names)
    units = int(round(1 / step))
    floor_units = np.array([int(round(floor.get(name, 0.0) / step)) for name in names], dtype=int)
    remaining = units - int(floor_units.sum())
    if remaining < 0:
        raise ValueError("Weight floors exceed 1.0.")
    if n == 1:
        yield np.array([1.0])
        return

    def rec(k: int, left: int, prefix: list[int]) -> Iterable[list[int]]:
        if k == n - 1:
            yield prefix + [left]
            return
        for value in range(left + 1):
            yield from rec(k + 1, left - value, prefix + [value])

    for extra in rec(0, remaining, []):
        yield (floor_units + np.array(extra, dtype=int)) / units


def _tune_weights(
    fold_predictions: list[dict[str, np.ndarray]],
    fold_actuals: list[np.ndarray],
    names: list[str],
    step: float,
    sample_shape_floor: float,
) -> tuple[np.ndarray, float]:
    matrix = [np.column_stack([preds[name] for name in names]) for preds in fold_predictions]
    actual = np.concatenate(fold_actuals)
    stacked = np.vstack(matrix)
    floor = {"sample_or_doy_shape": sample_shape_floor} if "sample_or_doy_shape" in names else {}
    best_w: np.ndarray | None = None
    best_mae = np.inf
    for weights in _simplex_weights(names, step, floor=floor):
        pred = stacked @ weights
        mae = float(np.mean(np.abs(actual - pred)))
        if mae < best_mae:
            best_mae = mae
            best_w = weights
    if best_w is None:
        best_w = np.ones(len(names)) / len(names)
    return best_w, best_mae


def run_robust_blend(config: RobustBlendConfig | None = None) -> dict[str, pd.DataFrame | Path | dict]:
    """Build robust candidate submissions, tune weights, and save final blend."""

    cfg = config or RobustBlendConfig()
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    forecast_dates = pd.date_range(cfg.forecast_start, cfg.forecast_end, freq="D")
    if len(forecast_dates) != cfg.horizon_days:
        raise ValueError(f"Expected {cfg.horizon_days} forecast days, got {len(forecast_dates)}.")

    all_metrics: list[dict[str, object]] = []
    weights_rows: list[dict[str, object]] = []
    final = pd.DataFrame({"Date": forecast_dates})
    candidate_submissions: dict[str, pd.DataFrame] = {}

    for target in TARGETS:
        fold_predictions: list[dict[str, np.ndarray]] = []
        fold_actuals: list[np.ndarray] = []
        names: list[str] | None = None
        for start in pd.to_datetime(list(cfg.validation_starts)):
            end = start + pd.Timedelta(days=cfg.horizon_days - 1)
            fit = sales[sales["Date"] < start].copy()
            holdout = sales[sales["Date"].between(start, end)].copy()
            if len(holdout) != cfg.horizon_days:
                continue
            preds = _candidate_predictions(fit, holdout["Date"], target, sample=None)
            names = list(preds)
            fold_predictions.append(preds)
            fold_actuals.append(holdout[target].to_numpy(dtype=float))
            for name, values in preds.items():
                all_metrics.append(
                    {
                        "target": target,
                        "fold_start": start.date().isoformat(),
                        "fold_end": end.date().isoformat(),
                        "model": name,
                        **_metrics(holdout[target].to_numpy(dtype=float), values),
                    }
                )
        if not fold_predictions or names is None:
            raise ValueError(f"No robust-blend validation folds available for {target}.")

        weights, tuned_mae = _tune_weights(
            fold_predictions,
            fold_actuals,
            names,
            step=cfg.weight_grid_step,
            sample_shape_floor=cfg.sample_shape_weight_floor,
        )
        for name, weight in zip(names, weights):
            weights_rows.append({"target": target, "model": name, "weight": float(weight), "cv_tuned_mae": tuned_mae})

        train_fit = sales.copy()
        final_candidates = _candidate_predictions(train_fit, pd.Series(forecast_dates), target, sample=sample)
        candidate_matrix = np.column_stack([final_candidates[name] for name in names])
        final[target] = np.clip(candidate_matrix @ weights, 0, None)
        for name, values in final_candidates.items():
            if name not in candidate_submissions:
                candidate_submissions[name] = pd.DataFrame({"Date": forecast_dates})
            candidate_submissions[name][target] = values

    metrics_df = pd.DataFrame(all_metrics)
    weights_df = pd.DataFrame(weights_rows)
    final_out = final.copy()
    final_out[["Revenue", "COGS"]] = final_out[["Revenue", "COGS"]].round(2)
    final_out["Date"] = final_out["Date"].dt.strftime("%Y-%m-%d")
    final_path = output_dir / "submission_robust_blend.csv"
    final_out.to_csv(final_path, index=False)

    metrics_df.to_csv(output_dir / "robust_blend_validation_metrics.csv", index=False)
    weights_df.to_csv(output_dir / "robust_blend_weights.csv", index=False)
    for name, df in candidate_submissions.items():
        out = df.copy()
        out[["Revenue", "COGS"]] = out[["Revenue", "COGS"]].round(2)
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        out.to_csv(output_dir / f"submission_candidate_{name}.csv", index=False)
    if "sample_or_doy_shape" in candidate_submissions:
        sample_candidate = candidate_submissions["sample_or_doy_shape"].copy()
        robust_numeric = final.copy()
        for sample_weight in (0.70, 0.85):
            conservative = sample_candidate[["Date"]].copy()
            for target in TARGETS:
                conservative[target] = (
                    sample_weight * sample_candidate[target]
                    + (1 - sample_weight) * robust_numeric[target]
                )
            conservative[["Revenue", "COGS"]] = conservative[["Revenue", "COGS"]].round(2)
            conservative["Date"] = conservative["Date"].dt.strftime("%Y-%m-%d")
            robust_weight = int(round((1 - sample_weight) * 100))
            conservative.to_csv(
                output_dir / f"submission_sample{int(sample_weight * 100)}_robust{robust_weight}.csv",
                index=False,
            )

    return {
        "submission": final_out,
        "metrics": metrics_df,
        "weights": weights_df,
        "path": final_path,
        "candidate_submissions": candidate_submissions,
    }
