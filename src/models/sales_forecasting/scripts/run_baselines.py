"""Run baseline models — SAME protocol as main pipeline:
recursive 548-day forecast, exogenous blanked after cutoff.

Outputs:
  docs/tables/baseline_results.csv
  docs/tables/pipeline_results.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.models.sales_forecasting import ARTIFACTS_DIR, DATA_DIR, DOCS_DIR

warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parents[2]
DATA = DATA_DIR
OUT = DOCS_DIR / "tables"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sales_forecasting.components.final_model import (
    FORECAST_END,
    FORECAST_START,
    add_calendar,
    add_lag_rolling,
    build_exogenous,
    feature_cols,
    _fill_exog_with_doy_history,
)

LEAKY_COLS = {
    "items_gross_value", "items_cogs_total_value",
    "pay_total_value", "pay_mean_value",
    "items_discount_total",
    "orders_count", "orders_unique_customers",
}

CV_FOLDS = [
    ("fold_2020_2021", pd.Timestamp("2020-06-30"), pd.date_range("2020-07-01", periods=548, freq="D")),
    ("fold_2021_2022", pd.Timestamp("2021-06-30"), pd.date_range("2021-07-01", periods=548, freq="D")),
]


def calc_metrics(actual, pred):
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
    }


def build_frame_with_cutoff(target, cutoff):
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    all_dates = pd.date_range(sales["Date"].min(), FORECAST_END, freq="D")

    exog = build_exogenous(all_dates)
    exog_cols = [c for c in exog.columns if c != "Date"]

    exog["doy_tmp"] = exog["Date"].dt.dayofyear
    hist_mask = exog["Date"] <= cutoff
    for col in exog_cols:
        doy_mean = exog.loc[hist_mask].groupby("doy_tmp")[col].mean()
        exog[col] = exog["doy_tmp"].map(doy_mean)
    exog = exog.drop(columns=["doy_tmp"])

    df = pd.DataFrame({"Date": all_dates}).merge(sales[["Date", target]], on="Date", how="left")
    df = df.merge(exog, on="Date", how="left")
    df = add_lag_rolling(df, target)
    df = add_calendar(df)
    return df


def recursive_forecast_sklearn(df, target, model, feats, forecast_idx):
    df = df.copy()
    for i in forecast_idx:
        tmp = df[["Date", target]].iloc[:i + 1].copy()
        tmp = add_lag_rolling(tmp, target)
        tmp = add_calendar(tmp)
        row = tmp.iloc[-1:].copy()
        exog_row = df.iloc[i:i + 1].drop(columns=[target])
        for c in exog_row.columns:
            if c not in row.columns:
                row[c] = exog_row[c].values
        for c in feats:
            if c not in row.columns:
                row[c] = np.nan
        yhat = max(0.0, float(model.predict(row[feats].fillna(0).values)[0]))
        df.iloc[i, df.columns.get_loc(target)] = yhat
    return df.iloc[forecast_idx][target].values


def run_baselines() -> pd.DataFrame:
    rows = []
    models = {
        "Linear Regression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=1.0),
        "Lasso": lambda: Lasso(alpha=1.0, max_iter=5000),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
        ),
    }

    for target in ["Revenue", "COGS"]:
        sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")

        for fold_name, cutoff, val_dates in CV_FOLDS:
            df = build_frame_with_cutoff(target, cutoff)
            feats = [f for f in feature_cols(df, target) if f not in LEAKY_COLS]

            df.loc[df["Date"] > cutoff, target] = np.nan
            train_mask = (df["Date"] >= "2014-01-01") & (df["Date"] <= cutoff)
            train = df[train_mask].dropna(subset=[target])
            X_tr = train[feats].fillna(0).values
            y_tr = train[target].values

            val_mask = df["Date"].isin(val_dates)
            actual = sales.set_index("Date").loc[val_dates, target].values
            forecast_idx = df.index[df["Date"].isin(val_dates)].values

            for name, factory in models.items():
                print(f"  {target} | {fold_name} | {name}...", end="", flush=True)
                m = factory()
                m.fit(X_tr, y_tr)
                pred = recursive_forecast_sklearn(df, target, m, feats, forecast_idx)
                met = calc_metrics(actual, pred)
                rows.append({"Target": target, "Model": name, "Fold": fold_name, **met})
                print(f" MAE={met['MAE']:>12,.0f}  R2={met['R2']:.4f}")

    df_out = pd.DataFrame(rows)
    summary = df_out.groupby(["Target", "Model"])[["MAE", "RMSE", "R2"]].mean().reset_index().round(4)
    return summary


def collect_pipeline_results() -> pd.DataFrame:
    rows = []
    cv_path = ARTIFACTS_DIR / "cv_metrics.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        label_map = {
            "cv_weighted_ensemble": "Recursive LGBM Ensemble (CV-weighted)",
            "model": "Recursive LGBM (9-model bag)",
            "seasonal": "Seasonal Naive (lag-364)",
            "doy": "DoY Climatology Mean",
        }
        for _, r in cv.iterrows():
            rows.append({
                "Target": r["target"],
                "Model": label_map.get(r["model"], r["model"]),
                "Fold": r["fold"],
                "MAE": round(r["mae"], 4),
                "RMSE": round(r["rmse"], 4),
                "R2": round(r["r2"], 4),
            })

    direct_path = ARTIFACTS_DIR / "direct_factory_cv_metrics.csv"
    if direct_path.exists():
        direct = pd.read_csv(direct_path)
        label_map = {
            "weighted_direct": "Direct LGBM+Ridge Ensemble",
            "lgb": "Direct LGBM Only",
            "ridge": "Direct Ridge Only",
            "doy_prior": "DoY Prior Baseline",
        }
        for _, r in direct.iterrows():
            rows.append({
                "Target": r["target"],
                "Model": label_map.get(r["model"], r["model"]),
                "Fold": r["fold_cutoff"],
                "MAE": round(r["mae"], 4),
                "RMSE": round(r["rmse"], 4),
                "R2": round(r["r2"], 4),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("Running baselines (recursive 548-day, exogenous blanked after cutoff)...\n")
    baseline = run_baselines()
    baseline.to_csv(OUT / "baseline_results.csv", index=False)
    print(f"\nbaseline_results.csv  ({len(baseline)} rows)")
    print(baseline.to_string(index=False))

    print("\nCollecting pipeline results...")
    pipeline = collect_pipeline_results()
    pipeline.to_csv(OUT / "pipeline_results.csv", index=False)
    print(f"pipeline_results.csv  ({len(pipeline)} rows)")

    print("\n\n========== FULL COMPARISON (548-day recursive, leakage-safe) ==========\n")
    all_rows = []
    for _, r in baseline.iterrows():
        all_rows.append({"Source": "Baseline", **r.to_dict()})
    avg_pipe = pipeline.groupby(["Target", "Model"])[["MAE", "RMSE", "R2"]].mean().reset_index().round(4)
    for _, r in avg_pipe.iterrows():
        all_rows.append({"Source": "Our Pipeline", **r.to_dict()})
    combined = pd.DataFrame(all_rows).sort_values(["Target", "R2"], ascending=[True, False])
    print(combined.to_string(index=False))
