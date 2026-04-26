"""Leak audit for v4 feature matrix.

Computes per-feature Pearson correlation with Revenue and COGS on the
training period and flags anything with |corr| > 0.95 as level-leak.

Also checks NaN rate on 2014-2022 and on the forecast horizon after
DoY-mean imputation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features_v4 import build_exog_v4, LEAKY_LEVEL_COLS_V4


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "final_v4"
OUT.mkdir(parents=True, exist_ok=True)

LEAK_THRESHOLD = 0.95


def doy_mean_impute(df: pd.DataFrame, cols: list[str], hist_end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["_doy"] = out["Date"].dt.dayofyear
    hist_mask = out["Date"] <= hist_end
    for c in cols:
        if c not in out.columns:
            continue
        m = out.loc[hist_mask].groupby("_doy")[c].mean()
        out[c] = out["_doy"].map(m).astype(float)
    out = out.drop(columns=["_doy"])
    return out


def audit() -> dict:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    all_dates = pd.date_range(sales["Date"].min(), pd.Timestamp("2024-07-01"), freq="D")
    exog = build_exog_v4(all_dates)

    # Merge target for corr calc on train rows
    train = exog.merge(sales, on="Date", how="left")
    train = train[train["Date"].between("2014-01-01", "2022-12-31")]

    feat_cols = [c for c in exog.columns if c != "Date"]

    rows = []
    for c in feat_cols:
        s = train[c]
        if s.dtype == "O":
            continue
        if s.std(skipna=True) == 0 or s.isna().all():
            rev_corr, cogs_corr = 0.0, 0.0
        else:
            rev_corr = float(np.corrcoef(
                s.fillna(s.median()), train["Revenue"].fillna(train["Revenue"].median())
            )[0, 1])
            cogs_corr = float(np.corrcoef(
                s.fillna(s.median()), train["COGS"].fillna(train["COGS"].median())
            )[0, 1])
        nan_rate_train = float(s.isna().mean())
        horizon_mask = exog["Date"] >= "2023-01-01"
        nan_rate_horizon = float(exog.loc[horizon_mask, c].isna().mean())
        rows.append({
            "feature": c,
            "corr_revenue": rev_corr,
            "corr_cogs": cogs_corr,
            "abs_max_corr": max(abs(rev_corr), abs(cogs_corr)),
            "nan_rate_train": nan_rate_train,
            "nan_rate_horizon_raw": nan_rate_horizon,
            "declared_level_leak": c in LEAKY_LEVEL_COLS_V4,
        })

    rep = pd.DataFrame(rows).sort_values("abs_max_corr", ascending=False)
    rep_path = OUT / "leak_audit.csv"
    rep.to_csv(rep_path, index=False)

    leaks = rep[(rep["abs_max_corr"] > LEAK_THRESHOLD) & (~rep["declared_level_leak"])]
    summary = {
        "total_features": int(len(rep)),
        "undeclared_leaks": leaks["feature"].tolist(),
        "declared_level_leaks": sorted(LEAKY_LEVEL_COLS_V4),
        "top5_by_corr": rep.head(5).to_dict(orient="records"),
        "report_path": str(rep_path),
    }

    # Sanity: after DoY-mean impute of level-leaks, horizon NaN should be ~0
    imputed = doy_mean_impute(exog, list(LEAKY_LEVEL_COLS_V4), pd.Timestamp("2022-12-31"))
    horizon = imputed[imputed["Date"] >= "2023-01-01"]
    horizon_nans = horizon.drop(columns=["Date"]).isna().mean().sort_values(ascending=False)
    summary["horizon_nan_top10_after_impute"] = horizon_nans.head(10).to_dict()

    with open(OUT / "leak_audit.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    s = audit()
    print(f"Total features: {s['total_features']}")
    print(f"Declared level leaks: {s['declared_level_leaks']}")
    print(f"Undeclared leaks (|corr| > {LEAK_THRESHOLD}): {s['undeclared_leaks']}")
    print("Top 5 by |corr|:")
    for r in s["top5_by_corr"]:
        print(f"  {r['feature']:40s}  rev={r['corr_revenue']:+.3f}  cogs={r['corr_cogs']:+.3f}")
    print("\nHorizon NaN top 10 after impute:")
    for k, v in list(s["horizon_nan_top10_after_impute"].items())[:10]:
        print(f"  {k:40s}  {v:.4f}")
