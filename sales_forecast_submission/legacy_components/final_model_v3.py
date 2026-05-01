"""Compliant LGBM v3 — Tweedie loss, deeper trees, alternative features.

Diversity pillar #3 to blend with v1 (legacy LGBM) and v2 (multi-seed).

Key differences vs v2:
  - Objective: tweedie (variance_power=1.6) — natural fit for skewed positive
    sales data; produces less spiky predictions.
  - Hyperparams: deeper trees (num_leaves=127), higher lr 0.04, no log target.
  - Features: drops doy_anchor lags (different bias), adds polynomial calendar
    interactions (year*sin_doy etc.).
  - Multi-seed bag (5 seeds).

Compliance: same as v2 — never reads sample_submission values.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from legacy_components.final_model_v2 import (
    LEAKY_EXO_COLS,
    DOY_MEAN_EXO_COLS,
    replace_with_doy_mean,
    add_calendar,
    build_exogenous,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
DATA = ROOT.parent / "data"
OUT = ROOT / "artifacts" / "legacy_component_outputs" / "final_v3"
OUT.mkdir(parents=True, exist_ok=True)

LAGS = (7, 14, 28, 56, 91, 182, 365, 730)
ROLLING = (7, 14, 28, 56, 91, 182, 365)

FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")


def add_lag_rolling_v3(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    s = df[col]
    for L in LAGS:
        df[f"{col}_lag{L}"] = s.shift(L)
    for W in ROLLING:
        shifted = s.shift(1)
        df[f"{col}_rmean{W}"] = shifted.rolling(W).mean()
        df[f"{col}_rmedian{W}"] = shifted.rolling(W).median()
        df[f"{col}_rmax{W}"] = shifted.rolling(W).max()
    # add lag*rolling interactions
    df[f"{col}_lag365_x_rmean28"] = df[f"{col}_lag365"] * df[f"{col}_rmean28"]
    return df


def build_frame_v3(target: str) -> pd.DataFrame:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    all_dates = pd.date_range(sales["Date"].min(), FORECAST_END, freq="D")
    exog = build_exogenous(all_dates)
    exog = exog.drop(columns=[c for c in LEAKY_EXO_COLS if c in exog.columns])
    exog = replace_with_doy_mean(exog, DOY_MEAN_EXO_COLS, hist_end=pd.Timestamp("2022-12-31"))

    df = pd.DataFrame({"Date": all_dates}).merge(
        sales[["Date", target]], on="Date", how="left"
    )
    df = df.merge(exog, on="Date", how="left")
    df = add_lag_rolling_v3(df, target)
    df = add_calendar(df)
    # polynomial calendar interactions
    df["sin_doy_x_year"] = df["sin_doy"] * df["year"]
    df["cos_doy_x_year"] = df["cos_doy"] * df["year"]
    df["sin_dow_x_year"] = df["sin_dow"] * df["year"]
    return df


EXCLUDE = {"Date"}


def feature_cols(df, target):
    return [c for c in df.columns if c not in EXCLUDE and c != target and df[c].dtype != "datetime64[ns]"]


def lgb_params_v3(seed: int) -> dict:
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.6,
        "metric": "mae",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 4,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
    }


def train_seed(train_df, val_df, feats, target, seed, n_round=4000):
    dtrain = lgb.Dataset(train_df[feats], label=train_df[target])
    dval = lgb.Dataset(val_df[feats], label=val_df[target], reference=dtrain)
    return lgb.train(
        lgb_params_v3(seed),
        dtrain,
        num_boost_round=n_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
    )


def train_full(full_df, feats, target, seed, n_round):
    d = lgb.Dataset(full_df[feats], label=full_df[target])
    return lgb.train(
        lgb_params_v3(seed), d, num_boost_round=n_round,
        valid_sets=[d], valid_names=["full"],
        callbacks=[lgb.log_evaluation(0)],
    )


def predict(models, X, feats):
    return np.mean([m.predict(X[feats]) for m in models], axis=0)


def recursive_forecast(df, target, models, feats):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    fidx = df.index[df[target].isna()].to_numpy()
    for i in fidx:
        tmp = df[["Date", target]].iloc[: i + 1].copy()
        tmp = add_lag_rolling_v3(tmp, target)
        tmp = add_calendar(tmp)
        tmp["sin_doy_x_year"] = tmp["sin_doy"] * tmp["year"]
        tmp["cos_doy_x_year"] = tmp["cos_doy"] * tmp["year"]
        tmp["sin_dow_x_year"] = tmp["sin_dow"] * tmp["year"]
        row = tmp.iloc[-1:].copy()
        exog_row = df.iloc[i : i + 1].drop(columns=[target])
        for c in exog_row.columns:
            if c not in row.columns:
                row[c] = exog_row[c].values
        for c in feats:
            if c not in row.columns:
                row[c] = np.nan
        yhat = float(predict(models, row, feats)[0])
        df.loc[i, target] = max(0.0, yhat)
    return df.loc[fidx, target].to_numpy()


def metrics(actual, pred):
    err = actual - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / max(1e-9, np.sum((actual - actual.mean()) ** 2))),
    }


def run(seeds=(42, 123, 7, 2024, 31)):
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    out_sub = pd.DataFrame({"Date": forecast_dates})
    results = {"seeds": list(seeds), "cv": {}, "feature_importance": {}}

    for target in ("Revenue", "COGS"):
        df = build_frame_v3(target)
        feats = feature_cols(df, target)
        hist = df.dropna(subset=[target])
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < "2022-01-01"]
        val = hist[hist["Date"].dt.year == 2022]

        cv_rows = []
        for s in seeds:
            m = train_seed(train, val, feats, target, s)
            p = m.predict(val[feats], num_iteration=m.best_iteration)
            cv_rows.append({"seed": s, "best_iter": m.best_iteration, **metrics(val[target].to_numpy(), p)})
        results["cv"][target] = cv_rows

        full_models = []
        for s in seeds:
            m_es = train_seed(train, val, feats, target, s)
            n = int((m_es.best_iteration or 1500) * 1.1)
            full_models.append(train_full(hist, feats, target, s, n))
        raw = recursive_forecast(df, target, full_models, feats)
        out_sub[target] = raw

        gains = np.mean([m.feature_importance(importance_type="gain") for m in full_models], axis=0)
        imp = pd.DataFrame({"feature": feats, "gain": gains}).sort_values("gain", ascending=False)
        imp.head(50).to_csv(OUT / f"feature_importance_{target.lower()}.csv", index=False)
        results["feature_importance"][target] = imp.head(15).to_dict(orient="records")
        for i, m in enumerate(full_models):
            m.save_model(str(OUT / f"lgbm_v3_{target.lower()}_seed{seeds[i]}.txt"))

    out_sub["Revenue"] = out_sub["Revenue"].round(2)
    out_sub["COGS"] = out_sub["COGS"].round(2)
    raw_path = OUT / "model_v3_raw.csv"
    cs = out_sub.copy()
    cs["Date"] = cs["Date"].dt.strftime("%Y-%m-%d")
    cs.to_csv(raw_path, index=False)
    results["file"] = str(raw_path)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {raw_path}")
    return results


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = run()
    print(json.dumps({k: v for k, v in r["cv"].items()}, indent=2))
    print(f"Elapsed {time.time()-t0:.1f}s")
