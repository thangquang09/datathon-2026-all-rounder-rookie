"""Local cross-validation emulating the public LB.

Strategy: holdout = last 548 days of train (~2021-07-03 -> 2022-12-31).
Use train data up to 2021-07-02 as history. Rebuild a forecast for that
holdout using the same methods we use for the real test, and compute
MAE / RMSE / R^2 separately for Revenue and COGS, plus a composite
LB-proxy score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "sales.csv"


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    mean_actual: float
    mean_pred: float


def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> Metrics:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return Metrics(mae=mae, rmse=rmse, r2=r2, mean_actual=float(actual.mean()), mean_pred=float(pred.mean()))


def load_holdout(horizon_days: int = 548) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(train)
    fit = train.iloc[: n - horizon_days].reset_index(drop=True)
    holdout = train.iloc[n - horizon_days :].reset_index(drop=True)
    return fit, holdout


def doy_average_forecast(
    fit: pd.DataFrame,
    target_dates: pd.Series,
    target: str,
    years_window: tuple[int, int],
    half_life_years: float = 0.0,
) -> np.ndarray:
    subset = fit[(fit["Date"].dt.year >= years_window[0]) & (fit["Date"].dt.year <= years_window[1])].copy()
    if half_life_years > 0:
        ref_year = subset["Date"].dt.year.max()
        subset["age"] = ref_year - subset["Date"].dt.year
        subset["weight"] = 0.5 ** (subset["age"] / max(half_life_years, 1e-6))
    else:
        subset["weight"] = 1.0
    subset["doy"] = subset["Date"].dt.dayofyear
    shape = subset.groupby("doy").apply(lambda g: np.average(g[target], weights=g["weight"]))
    doys = pd.to_datetime(target_dates).dt.dayofyear
    preds = doys.map(shape).to_numpy()
    return preds


def apply_per_year_scale(preds: np.ndarray, dates: pd.Series, scales: dict[int, float]) -> np.ndarray:
    out = preds.copy().astype(float)
    years = pd.to_datetime(dates).dt.year
    for y, s in scales.items():
        mask = (years == y).to_numpy()
        out[mask] = preds[mask] * s
    return out


def composite_lb_proxy(rev_m: Metrics, cog_m: Metrics) -> float:
    """A best-guess proxy for the combined LB metric.

    The LB we've observed is in the range 700k - 1.2M for our candidates.
    Our probes:
      - pure sample x 1.00: pred_mean ~3.1M (rev), actual unknown. MAE alone likely ~1M -> 1.22M
      - shape100 (eff 1.29/1.39): MAE ~500-600k likely -> 704k
    Reasonable proxy: MAE(Revenue) + 0.5 * RMSE(Revenue) + MAE(COGS)*0.8
    For now we use simple sum of MAEs as first-order proxy.
    """
    return rev_m.mae + cog_m.mae


def evaluate(
    pred_rev: np.ndarray,
    pred_cog: np.ndarray,
    holdout: pd.DataFrame,
) -> dict:
    actual_rev = holdout["Revenue"].to_numpy()
    actual_cog = holdout["COGS"].to_numpy()
    rev_m = compute_metrics(actual_rev, pred_rev)
    cog_m = compute_metrics(actual_cog, pred_cog)
    return {
        "revenue": rev_m.__dict__,
        "cogs": cog_m.__dict__,
        "lb_proxy": composite_lb_proxy(rev_m, cog_m),
    }


def run_doy_sweep(holdout_horizon: int = 548) -> pd.DataFrame:
    """Sweep (years_window, scale_2023, scale_2024) to find best local proxy."""
    fit, holdout = load_holdout(holdout_horizon)
    dates = holdout["Date"]

    results = []
    windows = [(2017, 2020), (2018, 2020), (2019, 2020), (2016, 2020), (2015, 2020)]
    # NOTE: fit ends at 2021-07-02 so max year in fit is 2021 (6 months).
    # Holdout covers 2021-07 through 2022-12.
    # We use 2019-2020 full years + partial 2021 for DoY shape.

    for window in windows:
        # Re-derive: we use years up to last full year before holdout start
        rev_shape_preds = doy_average_forecast(fit, dates, "Revenue", window)
        cog_shape_preds = doy_average_forecast(fit, dates, "COGS", window)

        for s_rev_23 in np.arange(1.00, 1.51, 0.05):
            for s_rev_24 in np.arange(1.00, 1.61, 0.05):
                # In the holdout 2021-07..2022-12, 2021 -> partial (July->Dec), 2022 -> full
                scales_rev = {2021: s_rev_23, 2022: s_rev_24}  # use 23/24 as proxy for 21/22
                scales_cog = {2021: s_rev_23, 2022: s_rev_24}  # mirror for COGS for this sweep
                rev_scaled = apply_per_year_scale(rev_shape_preds, dates, scales_rev)
                cog_scaled = apply_per_year_scale(cog_shape_preds, dates, scales_cog)
                m = evaluate(rev_scaled, cog_scaled, holdout)
                results.append({
                    "window": f"{window[0]}-{window[1]}",
                    "scale_year_early": s_rev_23,
                    "scale_year_late": s_rev_24,
                    "rev_mae": m["revenue"]["mae"],
                    "cog_mae": m["cogs"]["mae"],
                    "rev_rmse": m["revenue"]["rmse"],
                    "cog_rmse": m["cogs"]["rmse"],
                    "rev_r2": m["revenue"]["r2"],
                    "cog_r2": m["cogs"]["r2"],
                    "lb_proxy": m["lb_proxy"],
                })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_doy_sweep()
    df_sorted = df.sort_values("lb_proxy").reset_index(drop=True)
    print("Top 15 configs by lb_proxy:")
    print(df_sorted.head(15).to_string(index=False))
    print("\nBest per window:")
    best_per_win = df_sorted.groupby("window", as_index=False).first().sort_values("lb_proxy")
    print(best_per_win.to_string(index=False))
