"""v5 — LGBM with Huber (robust) objective for diversity.

Uses the exact same v4 feature pipeline (no-leak DoY-imputed exog + VN
calendar) but switches objective to Huber regression which is robust
to outliers (Tet spikes, BF, 11/11 peaks). Paired with a smaller num_leaves
to avoid overfitting.

Paired with v1/v2/v3/v4 in blends it adds a fresh error-direction signal.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.final_model_v4 import (
    HIST_END, FORECAST_START, FORECAST_END,
    build_frame, feature_cols, metrics,
    recursive_forecast, fit_train_yearly_trend, calibrate_levels,
    LB_LEVELS,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "final_v5"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 123, 7, 2024, 31)


def lgb_params_v5(seed: int) -> dict:
    return {
        "objective": "huber",
        "alpha": 0.9,
        "metric": "mae",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "lambda_l1": 0.3,
        "lambda_l2": 1.5,
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
    }


def _train_one_seed_v5(train_df, val_df, feats, target, seed, num_boost_round=5000):
    y_tr = np.log1p(train_df[target].to_numpy())
    y_vl = np.log1p(val_df[target].to_numpy())
    dtrain = lgb.Dataset(train_df[feats], label=y_tr)
    dval = lgb.Dataset(val_df[feats], label=y_vl, reference=dtrain)
    return lgb.train(
        lgb_params_v5(seed), dtrain, num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(250), lgb.log_evaluation(0)],
    )


def _train_full_seed_v5(full_df, feats, target, seed, num_boost_round):
    y = np.log1p(full_df[target].to_numpy())
    dtrain = lgb.Dataset(full_df[feats], label=y)
    return lgb.train(
        lgb_params_v5(seed), dtrain, num_boost_round=num_boost_round,
        valid_sets=[dtrain], valid_names=["train_full"],
        callbacks=[lgb.log_evaluation(0)],
    )


def walk_forward_cv(target: str, seeds=SEEDS) -> dict:
    df = build_frame(target)
    feats = feature_cols(df, target)
    rows = []
    for val_year in (2020, 2021, 2022):
        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < f"{val_year}-01-01"]
        val = hist[hist["Date"].dt.year == val_year]
        per_seed = []
        for s in seeds:
            m = _train_one_seed_v5(train, val, feats, target, seed=s)
            per_seed.append(m.predict(val[feats], num_iteration=m.best_iteration))
        log_pred = np.mean(per_seed, axis=0)
        pred = np.expm1(log_pred)
        actual = val[target].to_numpy()
        rows.append({"val_year": val_year, **metrics(actual, pred)})
    return {"target": target, "folds": rows, "n_features": len(feats)}


def fit_and_forecast(target: str, seeds=SEEDS):
    df = build_frame(target)
    feats = feature_cols(df, target)
    hist = df.dropna(subset=[target]).copy()
    hist = hist[hist["Date"] >= "2014-01-01"]
    train = hist[hist["Date"] < "2022-01-01"]
    val = hist[hist["Date"].dt.year == 2022]
    full_models = []
    for s in seeds:
        m_es = _train_one_seed_v5(train, val, feats, target, seed=s)
        best_iter = m_es.best_iteration or 2500
        m_full = _train_full_seed_v5(hist, feats, target, seed=s,
                                     num_boost_round=int(best_iter * 1.1))
        full_models.append(m_full)

    # Recursive forecast — use v4's helper which already handles log1p
    from src.final_model_v4 import predict_log
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    from src.calendar_vn import add_vn_calendar
    from src.final_model import add_calendar as v_calendar, add_lag_rolling
    forecast_idx = df2.index[df2[target].isna()].to_numpy()
    for i in forecast_idx:
        tmp = df2[["Date", target]].iloc[: i + 1].copy()
        tmp = add_lag_rolling(tmp, target)
        tmp = v_calendar(tmp)
        tmp = add_vn_calendar(tmp)
        row = tmp.iloc[-1:].copy()
        exog_row = df2.iloc[i : i + 1].drop(columns=[target])
        for c in exog_row.columns:
            if c not in row.columns:
                row[c] = exog_row[c].values
        for c in feats:
            if c not in row.columns:
                row[c] = np.nan
        log_pred = predict_log(full_models, row, feats)[0]
        yhat = max(0.0, float(np.expm1(log_pred)))
        df2.loc[i, target] = yhat
    raw = df2.loc[forecast_idx, target].to_numpy()
    return raw, full_models, df, feats


def run(seeds=SEEDS) -> dict:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")

    results = {"seeds": list(seeds), "cv": {}, "levels": {}}
    sub_raw = pd.DataFrame({"Date": forecast_dates})
    sub_lb = pd.DataFrame({"Date": forecast_dates})

    for target in ["Revenue", "COGS"]:
        cv = walk_forward_cv(target, seeds=seeds)
        results["cv"][target] = cv
        print(f"\nWF-CV {target}:  (n_features={cv['n_features']})")
        for r in cv["folds"]:
            print(f"  {r['val_year']}  MAE={r['mae']:>14,.0f}  RMSE={r['rmse']:>14,.0f}  R2={r['r2']:.4f}")

        raw, full_models, df, feats = fit_and_forecast(target, seeds=seeds)
        sub_raw[target] = raw
        sub_lb[target] = calibrate_levels(raw, forecast_dates, LB_LEVELS[target])

        for i, m in enumerate(full_models):
            m.save_model(str(OUT / f"lgbm_{target.lower()}_seed{seeds[i]}.txt"))

    for s in (sub_raw, sub_lb):
        s["Revenue"] = s["Revenue"].round(2)
        s["COGS"] = s["COGS"].round(2)
        assert len(s) == 548
        assert (s[["Revenue", "COGS"]] > 0).all().all()

    raw_path = OUT / "model_v5_raw.csv"
    lb_path = OUT / "model_v5_lb.csv"
    for s, p in [(sub_raw, raw_path), (sub_lb, lb_path)]:
        cs = s.copy()
        cs["Date"] = cs["Date"].dt.strftime("%Y-%m-%d")
        cs.to_csv(p, index=False)
    results["files"] = {"raw": str(raw_path), "lb": str(lb_path)}

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSubmissions: {results['files']}")
    return results


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\nElapsed {time.time() - t0:.1f}s")
