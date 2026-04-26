"""Chronos-2 zero-shot forecast for Revenue and COGS.

Chronos-2 (Amazon, Oct 2025) is a 120M-param encoder-only time series
foundation model that supports univariate, multivariate, and
covariate-informed forecasting in a zero-shot manner — no training
needed on our data.

Pipeline:
  1. Load sales.csv (train-only).
  2. For each target (Revenue, COGS), build a context dataframe in
     Chronos `predict_df` format.
  3. Run zero-shot forecast for 548 days ahead.
  4. Save raw forecast CSV for blending.

Compliance:
  - Only reads sales.csv (train-only, ends 2022-12-31).
  - No test labels, no sample_submission values.
  - Chronos-2 is a GENERAL foundation model pre-trained on PUBLIC
    datasets (fev-bench/GIFT-Eval) that do NOT include this
    competition's labels. No leakage.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from chronos import BaseChronosPipeline, Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "chronos"
OUT.mkdir(parents=True, exist_ok=True)


FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")


def make_context(sales: pd.DataFrame, target: str) -> pd.DataFrame:
    """Chronos predict_df expects columns: item_id, timestamp, target, and optional covariates."""
    df = sales[["Date", target]].copy()
    df = df.rename(columns={"Date": "timestamp", target: "target"})
    df["item_id"] = target
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["item_id", "timestamp", "target"]]


def run_chronos(seed: int = 42) -> pd.DataFrame:
    import torch

    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    prediction_length = len(forecast_dates)
    assert prediction_length == 548

    print(f"Loading Chronos-2 (CPU)...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    out = pd.DataFrame({"Date": forecast_dates})
    for target in ("Revenue", "COGS"):
        print(f"\nForecasting {target} ...")
        ctx = make_context(sales, target)
        pred = pipeline.predict_df(
            ctx,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        # pred is a DataFrame with columns item_id, timestamp, 0.1, 0.5, 0.9, mean
        pred_sorted = pred.sort_values("timestamp").reset_index(drop=True)
        assert len(pred_sorted) == prediction_length
        # Use 0.5 (median) as the point forecast (robust to heavy tails)
        out[target] = pred_sorted["0.5"].clip(lower=0.0).to_numpy()

    csv_path = OUT / "chronos2_raw.csv"
    cs = out.copy()
    cs["Date"] = cs["Date"].dt.strftime("%Y-%m-%d")
    cs["Revenue"] = cs["Revenue"].round(2)
    cs["COGS"] = cs["COGS"].round(2)
    cs.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    yearly = out.copy()
    yearly["year"] = yearly["Date"].dt.year
    print(yearly.groupby("year")[["Revenue", "COGS"]].mean().round(0))
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    run_chronos()
    print(f"\nElapsed {time.time() - t0:.1f}s")
