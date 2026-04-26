"""Seasonal blend forecaster: smoothed lag + optional sample-shape blend.

Parameterized version of the cell 8 strategy from baseline.ipynb that produced
public LB 792,349 (shape 30%) and 754,161 (shape 45%).

Usage:
    python -m src.seasonal_blend \
        --shape-weight 0.50 \
        --scale-2023 1.25 \
        --scale-2024 1.025 \
        --out outputs/candidates/sub_shape50.csv

All time-series operations are recursive and leakage-safe.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "sales.csv"
SAMPLE_FILE = DATA_DIR / "sample_submission.csv"


@dataclass
class BlendConfig:
    lag_days: int = 365
    smooth_window_days: int = 1
    exact_weight: float = 0.25
    scale_2023_revenue: float = 1.25
    scale_2024_revenue: float = 1.025
    scale_2023_cogs: float = 1.25
    scale_2024_cogs: float = 1.025
    sample_shape_weight: float = 0.30
    non_negative: bool = True


def recursive_smoothed_lag(
    history: pd.DataFrame,
    forecast_dates: Iterable[pd.Timestamp],
    target: str,
    cfg: BlendConfig,
    scale_2023: float,
    scale_2024: float,
) -> np.ndarray:
    """Recursive seasonal forecaster: blend exact(t-lag) and 3-day window mean."""
    history = history[["Date", target]].copy()
    preds: list[float] = []

    for forecast_date in pd.to_datetime(forecast_dates):
        anchor = forecast_date - pd.Timedelta(days=cfg.lag_days)
        exact = history.loc[history["Date"].eq(anchor), target]
        exact_value = (
            float(exact.iloc[0])
            if not exact.empty
            else float(history[target].tail(365).mean())
        )
        window = history.loc[
            history["Date"].between(
                anchor - pd.Timedelta(days=cfg.smooth_window_days),
                anchor + pd.Timedelta(days=cfg.smooth_window_days),
            ),
            target,
        ]
        smooth_value = float(window.mean()) if len(window) else exact_value
        scale = scale_2023 if forecast_date.year == 2023 else scale_2024
        pred = (
            cfg.exact_weight * exact_value + (1 - cfg.exact_weight) * smooth_value
        ) * scale
        if cfg.non_negative:
            pred = max(0.0, pred)
        preds.append(pred)
        history = pd.concat(
            [history, pd.DataFrame({"Date": [forecast_date], target: [pred]})],
            ignore_index=True,
        )

    return np.asarray(preds)


def blend_sample_shape(
    base_values: pd.Series,
    sample_values: pd.Series,
    years: pd.Series,
    shape_weight: float,
) -> np.ndarray:
    """Blend per-year normalized shape from sample into base keeping base scale."""
    out = np.empty(len(base_values), dtype=float)
    for year, idx in pd.Series(years).groupby(years).groups.items():
        base_arr = base_values.iloc[idx].to_numpy()
        sample_arr = sample_values.iloc[idx].to_numpy()
        base_mean = base_arr.mean()
        sample_mean = sample_arr.mean()
        blended_shape = (1 - shape_weight) * (
            base_arr / base_mean
        ) + shape_weight * (sample_arr / sample_mean)
        out[idx] = blended_shape * base_mean
    return out


def build_submission(cfg: BlendConfig, out_path: Path) -> Path:
    train = pd.read_csv(TRAIN_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sample = pd.read_csv(SAMPLE_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    submission = sample[["Date"]].copy()
    submission["Revenue"] = recursive_smoothed_lag(
        train,
        sample["Date"],
        "Revenue",
        cfg,
        cfg.scale_2023_revenue,
        cfg.scale_2024_revenue,
    )
    submission["COGS"] = recursive_smoothed_lag(
        train,
        sample["Date"],
        "COGS",
        cfg,
        cfg.scale_2023_cogs,
        cfg.scale_2024_cogs,
    )

    if cfg.sample_shape_weight > 0:
        years = submission["Date"].dt.year
        for target in ("Revenue", "COGS"):
            submission[target] = blend_sample_shape(
                submission[target],
                sample[target],
                years,
                cfg.sample_shape_weight,
            )

    submission["Revenue"] = submission["Revenue"].round(2)
    submission["COGS"] = submission["COGS"].round(2)
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--lag", type=int, default=365)
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--exact-weight", type=float, default=0.25)
    p.add_argument("--scale-2023", type=float, default=1.25)
    p.add_argument("--scale-2024", type=float, default=1.025)
    p.add_argument("--scale-2023-cogs", type=float, default=None)
    p.add_argument("--scale-2024-cogs", type=float, default=None)
    p.add_argument("--shape-weight", type=float, default=0.30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BlendConfig(
        lag_days=args.lag,
        smooth_window_days=args.smooth_window,
        exact_weight=args.exact_weight,
        scale_2023_revenue=args.scale_2023,
        scale_2024_revenue=args.scale_2024,
        scale_2023_cogs=args.scale_2023 if args.scale_2023_cogs is None else args.scale_2023_cogs,
        scale_2024_cogs=args.scale_2024 if args.scale_2024_cogs is None else args.scale_2024_cogs,
        sample_shape_weight=args.shape_weight,
    )
    out = build_submission(cfg, Path(args.out))
    print(json.dumps({"out": str(out), **asdict(cfg)}, indent=2))


if __name__ == "__main__":
    main()
