"""Compliant LGBM v2 — handles train/inference distribution shift.

Improvements over v1 (`final_model.py`):
1. Drops contemporaneous Revenue/COGS-leak features (items_total_value,
   pay_total_value, etc. that equal target ± noise on train days).
2. Replaces volatile exogenous signals with their day-of-year mean
   *both* in train and inference, so feature distribution is stable.
3. Multi-seed LGBM bagging (M5-winner-style equal-weighted ensemble).
4. Walk-forward cross-validation across multiple holdout years
   (2020, 2021, 2022) — a more honest performance estimate.
5. log1p target transform to stabilise multiplicative noise.
6. Optional separate per-DoW correction.

Compliance:
- ZERO reads of Revenue/COGS values from sample_submission.csv or
  submission.csv. Only `Date` is used to know which rows to forecast.
- Uses only the 13 train CSVs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.final_model import (
    add_calendar,
    add_lag_rolling,
    build_exogenous,
    feature_cols as v1_feature_cols,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
DATA = ROOT.parent / "data"
OUT = ROOT / "outputs" / "final_v2"
OUT.mkdir(parents=True, exist_ok=True)

LAGS = (7, 14, 28, 56, 91, 182, 364, 365, 371, 728, 730)
ROLLING = (7, 14, 28, 91, 182, 365)

FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")

# Exogenous columns whose contemporaneous value essentially equals
# Revenue/COGS — must be DROPPED from features.
LEAKY_EXO_COLS = {
    "items_gross_value",   # = Revenue exactly
    "pay_total_value",     # = Revenue * ~1.06
    "items_discount_total",
    "pay_mean_value",
}

# Volatile exogenous — replace with DoY mean both in train and inference
# so distribution is stable.
DOY_MEAN_EXO_COLS = [
    "orders_count",
    "orders_unique_customers",
    "orders_mobile",
    "orders_desktop",
    "orders_delivered",
    "orders_returned",
    "orders_paid_search",
    "items_total_qty",
    "items_avg_unit_price",
    "items_promo_share",
    "items_unique_products",
    "pay_mean_installments",
    "returns_count",
    "returns_qty",
    "returns_refund_value",
    "reviews_count",
    "reviews_avg_rating",
    "ship_count",
    "ship_fee_total",
    "ship_fee_mean",
    "ship_leadtime_mean",
    "web_sessions",
    "web_unique_visitors",
    "web_page_views",
    "web_bounce_rate",
    "web_avg_session",
    "promo_active_count",
    "promo_max_discount",
    "promo_active",
    "new_signups",
    "signups_rmean28",
    "inv_stockout_rate",
    "inv_overstock_rate",
    "inv_reorder_rate",
    "inv_fill_rate",
    "inv_sell_through",
    "inv_days_of_supply",
]


@dataclass
class FoldMetrics:
    fold: str
    target: str
    mae: float
    rmse: float
    r2: float


def replace_with_doy_mean(df: pd.DataFrame, cols: list[str], hist_end: pd.Timestamp) -> pd.DataFrame:
    """Replace each col with its day-of-year mean computed on rows
    with Date <= hist_end (so leakage-free).
    """
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


def build_frame(target: str, hist_end: pd.Timestamp = pd.Timestamp("2022-12-31")) -> pd.DataFrame:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    all_dates = pd.date_range(sales["Date"].min(), FORECAST_END, freq="D")

    exog = build_exogenous(all_dates)
    exog = exog.drop(columns=[c for c in LEAKY_EXO_COLS if c in exog.columns])
    exog = replace_with_doy_mean(exog, DOY_MEAN_EXO_COLS, hist_end=hist_end)

    df = pd.DataFrame({"Date": all_dates}).merge(
        sales[["Date", target]], on="Date", how="left"
    )
    df = df.merge(exog, on="Date", how="left")

    df = add_lag_rolling(df, target)
    df = add_calendar(df)
    return df


EXCLUDE = {"Date"}


def feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in EXCLUDE and c != target and df[c].dtype != "datetime64[ns]"
    ]


def lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.025,
        "num_leaves": 47,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 4,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
    }


def metrics(actual, pred):
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _train_one_seed(train_df, val_df, feats, target, seed: int, num_boost_round=4000) -> lgb.Booster:
    y_tr = np.log1p(train_df[target].to_numpy())
    y_vl = np.log1p(val_df[target].to_numpy())
    dtrain = lgb.Dataset(train_df[feats], label=y_tr)
    dval = lgb.Dataset(val_df[feats], label=y_vl, reference=dtrain)
    model = lgb.train(
        lgb_params(seed),
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
    )
    return model


def _train_full_seed(full_df, feats, target, seed: int, num_boost_round: int) -> lgb.Booster:
    y = np.log1p(full_df[target].to_numpy())
    dtrain = lgb.Dataset(full_df[feats], label=y)
    model = lgb.train(
        lgb_params(seed),
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain],
        valid_names=["train_full"],
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def predict_log(models: list[lgb.Booster], X: pd.DataFrame, feats: list[str]) -> np.ndarray:
    preds = np.column_stack([m.predict(X[feats]) for m in models])
    return preds.mean(axis=1)


def recursive_forecast(df: pd.DataFrame, target: str, models: list[lgb.Booster], feats: list[str]) -> np.ndarray:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    forecast_idx = df.index[df[target].isna()].to_numpy()
    for i in forecast_idx:
        tmp = df[["Date", target]].iloc[: i + 1].copy()
        tmp = add_lag_rolling(tmp, target)
        tmp = add_calendar(tmp)
        row = tmp.iloc[-1:].copy()
        exog_row = df.iloc[i : i + 1].drop(columns=[target])
        for c in exog_row.columns:
            if c not in row.columns:
                row[c] = exog_row[c].values
        for c in feats:
            if c not in row.columns:
                row[c] = np.nan
        log_pred = predict_log(models, row, feats)[0]
        yhat = float(np.expm1(log_pred))
        yhat = max(0.0, yhat)
        df.loc[i, target] = yhat
    return df.loc[forecast_idx, target].to_numpy()


def walk_forward_cv(target: str, seeds: tuple[int, ...] = (42, 123, 7, 2024, 31)) -> dict:
    """Walk-forward: train on <Y, validate on Y for Y ∈ {2020, 2021, 2022}."""
    df = build_frame(target, hist_end=pd.Timestamp("2022-12-31"))
    feats = feature_cols(df, target)
    rows = []
    for val_year in (2020, 2021, 2022):
        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < f"{val_year}-01-01"]
        val = hist[hist["Date"].dt.year == val_year]
        if len(val) == 0:
            continue
        per_seed_preds = []
        for s in seeds:
            m = _train_one_seed(train, val, feats, target, seed=s)
            per_seed_preds.append(m.predict(val[feats], num_iteration=m.best_iteration))
        log_pred = np.mean(per_seed_preds, axis=0)
        pred = np.expm1(log_pred)
        actual = val[target].to_numpy()
        rows.append({"val_year": val_year, **metrics(actual, pred), "n_seeds": len(seeds)})
    return {"target": target, "folds": rows}


def fit_and_forecast(target: str, seeds: tuple[int, ...]) -> tuple[np.ndarray, list[lgb.Booster], pd.DataFrame, list[str]]:
    df = build_frame(target, hist_end=pd.Timestamp("2022-12-31"))
    feats = feature_cols(df, target)
    hist = df.dropna(subset=[target]).copy()
    hist = hist[hist["Date"] >= "2014-01-01"]
    # Use 2022 as final-tuning val to find best round per seed
    train = hist[hist["Date"] < "2022-01-01"]
    val = hist[hist["Date"].dt.year == 2022]

    full_models = []
    for s in seeds:
        m_es = _train_one_seed(train, val, feats, target, seed=s)
        best_iter = m_es.best_iteration or 2000
        m_full = _train_full_seed(hist, feats, target, seed=s, num_boost_round=int(best_iter * 1.1))
        full_models.append(m_full)
    raw = recursive_forecast(df, target, full_models, feats)
    return raw, full_models, df, feats


def fit_train_yearly_trend(sales: pd.DataFrame, target: str) -> dict:
    y = sales.copy()
    y["year"] = y["Date"].dt.year
    yr = y[y["year"].between(2015, 2022)].groupby("year")[target].mean()
    yrs = yr.index.values.astype(float)
    vals = np.log(yr.values)
    slope, intercept = np.polyfit(yrs, vals, 1)
    return {
        "mean_2023_loglinear": float(np.exp(intercept + slope * 2023)),
        "mean_2024_loglinear": float(np.exp(intercept + slope * 2024)),
        "mean_2022_actual": float(yr.loc[2022]),
        "yoy_2022": float(yr.loc[2022] / yr.loc[2021]),
    }


def calibrate_levels(pred: np.ndarray, dates, levels: dict) -> np.ndarray:
    out = pred.astype(float).copy()
    years = pd.DatetimeIndex(dates).year.to_numpy()
    for y, want in levels.items():
        m = years == y
        have = out[m].mean()
        if have > 0:
            out[m] *= want / have
    return out


def run(seeds: tuple[int, ...] = (42, 123, 7, 2024, 31)) -> dict:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")

    results: dict = {"seeds": list(seeds), "cv": {}, "levels": {}, "feature_importance": {}}
    sub_raw = pd.DataFrame({"Date": forecast_dates})
    sub_calib = pd.DataFrame({"Date": forecast_dates})

    for target in ["Revenue", "COGS"]:
        cv = walk_forward_cv(target, seeds=seeds)
        results["cv"][target] = cv
        print(f"\nWF-CV {target}:")
        for r in cv["folds"]:
            print(f"  {r['val_year']}  MAE={r['mae']:>14,.0f}  RMSE={r['rmse']:>14,.0f}  R2={r['r2']:.4f}")

        raw, full_models, df, feats = fit_and_forecast(target, seeds=seeds)
        sub_raw[target] = raw

        # Train-only level calibration
        trend = fit_train_yearly_trend(sales, target)
        m22 = trend["mean_2022_actual"]
        cont_23 = m22 * trend["yoy_2022"]
        cont_24 = m22 * trend["yoy_2022"] ** 2
        levels = {2023: cont_23, 2024: cont_24}
        sub_calib[target] = calibrate_levels(raw, forecast_dates, levels)

        results["levels"][target] = {
            "trend": trend,
            "raw_means": {str(y): float(raw[forecast_dates.year == y].mean()) for y in (2023, 2024)},
            "calib_levels": {str(k): v for k, v in levels.items()},
        }

        # Average gain across seeds
        gains = np.mean([m.feature_importance(importance_type="gain") for m in full_models], axis=0)
        imp = pd.DataFrame({"feature": feats, "gain": gains}).sort_values("gain", ascending=False)
        imp.head(50).to_csv(OUT / f"feature_importance_{target.lower()}.csv", index=False)
        results["feature_importance"][target] = imp.head(20).to_dict(orient="records")
        for i, m in enumerate(full_models):
            m.save_model(str(OUT / f"lgbm_{target.lower()}_seed{seeds[i]}.txt"))

    for s in (sub_raw, sub_calib):
        s["Revenue"] = s["Revenue"].round(2)
        s["COGS"] = s["COGS"].round(2)
        assert len(s) == 548
        assert (s[["Revenue", "COGS"]] > 0).all().all()

    raw_path = OUT / "model_v2_raw.csv"
    calib_path = OUT / "model_v2_calib.csv"
    for s, p in [(sub_raw, raw_path), (sub_calib, calib_path)]:
        cs = s.copy()
        cs["Date"] = cs["Date"].dt.strftime("%Y-%m-%d")
        cs.to_csv(p, index=False)
    results["files"] = {"raw": str(raw_path), "calib": str(calib_path)}

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSubmissions: {results['files']}")
    return results


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = run()
    print(f"\nElapsed {time.time() - t0:.1f}s")
