"""Compliant final model for DATATHON 2026 Round 1 — Part 3.

Strict rules followed:
- Uses ONLY the 14 CSV files provided, but NEVER reads values from
  `sample_submission.csv` or `submission.csv`.
- Revenue/COGS are used only from `sales.csv` (the training split that
  ends on 2022-12-31). The forecast horizon 2023-01-01..2024-07-01
  NEVER reads Revenue/COGS from any test artifact.
- All exogenous signals (orders, order_items, shipments, returns,
  reviews, web_traffic, promotions, payments, inventory, customers,
  products, geography) are computed from train files. For the forecast
  horizon, per-date exogenous values are replaced by the historical
  day-of-year average so that no post-2022 information leaks in.
- Post-hoc calibration uses ONLY `sales.csv` yearly aggregates to fit a
  log-linear year trend; it does NOT touch sample_submission.

Outputs:
- artifacts/legacy_component_outputs/final/model_submission.csv
- artifacts/legacy_component_outputs/final/metrics.json
- artifacts/legacy_component_outputs/final/feature_importance_*.csv
- artifacts/legacy_component_outputs/final/shap_*.png
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.sales_forecasting import ARTIFACTS_DIR, DATA_DIR


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parents[2]
DATA = DATA_DIR
OUT = ARTIFACTS_DIR / "legacy_component_outputs" / "final"
OUT.mkdir(parents=True, exist_ok=True)

LAGS = (7, 14, 28, 56, 91, 182, 364, 365, 371, 728, 730)
ROLLING = (7, 14, 28, 91, 182, 365)

FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")


@dataclass
class FoldMetrics:
    split: str
    target: str
    mae: float
    rmse: float
    r2: float


# ---------------------------------------------------------------------------
# Exogenous aggregations from the 13 allowed CSVs (sales_test/sample not touched)
# ---------------------------------------------------------------------------


def _agg_orders() -> pd.DataFrame:
    df = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    g = df.groupby("order_date").agg(
        orders_count=("order_id", "count"),
        orders_unique_customers=("customer_id", "nunique"),
        orders_mobile=("device_type", lambda s: (s == "mobile").sum()),
        orders_desktop=("device_type", lambda s: (s == "desktop").sum()),
        orders_delivered=("order_status", lambda s: (s == "delivered").sum()),
        orders_returned=("order_status", lambda s: (s == "returned").sum()),
        orders_paid_search=("order_source", lambda s: (s == "paid_search").sum()),
    ).reset_index().rename(columns={"order_date": "Date"})
    return g


def _agg_order_items() -> pd.DataFrame:
    items = pd.read_csv(DATA / "order_items.csv")
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])[["order_id", "order_date"]]
    items = items.merge(orders, on="order_id", how="left")
    items["gross"] = items["quantity"] * items["unit_price"]
    items["has_promo"] = items["promo_id"].notna().astype(int)
    g = items.groupby("order_date").agg(
        items_total_qty=("quantity", "sum"),
        items_gross_value=("gross", "sum"),
        items_discount_total=("discount_amount", "sum"),
        items_avg_unit_price=("unit_price", "mean"),
        items_promo_share=("has_promo", "mean"),
        items_unique_products=("product_id", "nunique"),
    ).reset_index().rename(columns={"order_date": "Date"})
    return g


def _agg_payments() -> pd.DataFrame:
    pay = pd.read_csv(DATA / "payments.csv")
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])[["order_id", "order_date"]]
    pay = pay.merge(orders, on="order_id", how="left")
    g = pay.groupby("order_date").agg(
        pay_total_value=("payment_value", "sum"),
        pay_mean_value=("payment_value", "mean"),
        pay_mean_installments=("installments", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})
    return g


def _agg_returns() -> pd.DataFrame:
    ret = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
    g = ret.groupby("return_date").agg(
        returns_count=("return_id", "count"),
        returns_qty=("return_quantity", "sum"),
        returns_refund_value=("refund_amount", "sum"),
    ).reset_index().rename(columns={"return_date": "Date"})
    return g


def _agg_reviews() -> pd.DataFrame:
    rev = pd.read_csv(DATA / "reviews.csv", parse_dates=["review_date"])
    g = rev.groupby("review_date").agg(
        reviews_count=("review_id", "count"),
        reviews_avg_rating=("rating", "mean"),
    ).reset_index().rename(columns={"review_date": "Date"})
    return g


def _agg_shipments() -> pd.DataFrame:
    sh = pd.read_csv(DATA / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
    sh["leadtime"] = (sh["delivery_date"] - sh["ship_date"]).dt.days
    g = sh.groupby("ship_date").agg(
        ship_count=("order_id", "count"),
        ship_fee_total=("shipping_fee", "sum"),
        ship_fee_mean=("shipping_fee", "mean"),
        ship_leadtime_mean=("leadtime", "mean"),
    ).reset_index().rename(columns={"ship_date": "Date"})
    return g


def _agg_web_traffic() -> pd.DataFrame:
    web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
    g = web.groupby("date").agg(
        web_sessions=("sessions", "sum"),
        web_unique_visitors=("unique_visitors", "sum"),
        web_page_views=("page_views", "sum"),
        web_bounce_rate=("bounce_rate", "mean"),
        web_avg_session=("avg_session_duration_sec", "mean"),
    ).reset_index().rename(columns={"date": "Date"})
    return g


def _agg_promotions(all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    promos = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    out = pd.DataFrame({"Date": all_dates})
    out["promo_active_count"] = 0
    out["promo_max_discount"] = 0.0
    for _, r in promos.iterrows():
        mask = out["Date"].between(r["start_date"], r["end_date"])
        out.loc[mask, "promo_active_count"] += 1
        out.loc[mask, "promo_max_discount"] = np.maximum(
            out.loc[mask, "promo_max_discount"], float(r["discount_value"])
        )
    out["promo_active"] = (out["promo_active_count"] > 0).astype(int)
    return out


def _agg_customers(all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    cust = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    daily = cust.groupby("signup_date").size().rename("new_signups").reset_index().rename(
        columns={"signup_date": "Date"}
    )
    out = pd.DataFrame({"Date": all_dates}).merge(daily, on="Date", how="left")
    out["new_signups"] = out["new_signups"].fillna(0)
    out["signups_rmean28"] = out["new_signups"].rolling(28, min_periods=1).mean()
    return out


def _agg_inventory(all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    inv = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
    monthly = inv.groupby("snapshot_date").agg(
        inv_stockout_rate=("stockout_flag", "mean"),
        inv_overstock_rate=("overstock_flag", "mean"),
        inv_reorder_rate=("reorder_flag", "mean"),
        inv_fill_rate=("fill_rate", "mean"),
        inv_sell_through=("sell_through_rate", "mean"),
        inv_days_of_supply=("days_of_supply", "mean"),
    ).reset_index().rename(columns={"snapshot_date": "Date"})
    out = pd.DataFrame({"Date": all_dates}).merge(monthly, on="Date", how="left")
    # forward-fill monthly snapshot to daily
    cols = [c for c in out.columns if c != "Date"]
    out[cols] = out[cols].ffill()
    return out


def build_exogenous(all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    base = pd.DataFrame({"Date": all_dates})
    for f in (
        _agg_orders,
        _agg_order_items,
        _agg_payments,
        _agg_returns,
        _agg_reviews,
        _agg_shipments,
        _agg_web_traffic,
    ):
        base = base.merge(f(), on="Date", how="left")
    base = base.merge(_agg_promotions(all_dates), on="Date", how="left")
    base = base.merge(_agg_customers(all_dates), on="Date", how="left")
    base = base.merge(_agg_inventory(all_dates), on="Date", how="left")
    return base


# ---------------------------------------------------------------------------
# Calendar / lag / rolling
# ---------------------------------------------------------------------------


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["Date"]
    df = df.copy()
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["day"] = d.dt.day
    df["quarter"] = d.dt.quarter
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["year_trend"] = df["year"] - 2012
    df["days_since_start"] = (d - d.min()).dt.days
    return df


def add_lag_rolling(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    s = df[col]
    for L in LAGS:
        df[f"{col}_lag{L}"] = s.shift(L)
    for W in ROLLING:
        shifted = s.shift(1)
        df[f"{col}_rmean{W}"] = shifted.rolling(W).mean()
        df[f"{col}_rstd{W}"] = shifted.rolling(W).std()
    for L in (365, 730):
        df[f"{col}_doy_anchor_{L}"] = s.shift(L - 3).rolling(7).mean()
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _fill_exog_with_doy_history(exog: pd.DataFrame) -> pd.DataFrame:
    """Impute missing exogenous values with DoY mean from pre-2023 history.

    This is train-only information and DOES NOT leak future data.
    """
    out = exog.copy()
    out["doy_tmp"] = out["Date"].dt.dayofyear
    hist_mask = out["Date"].dt.year < 2023
    cols = [c for c in out.columns if c not in ("Date", "doy_tmp")]
    for col in cols:
        doy_mean = out.loc[hist_mask].groupby("doy_tmp")[col].mean()
        mask = out[col].isna()
        out.loc[mask, col] = out.loc[mask, "doy_tmp"].map(doy_mean)
    out = out.drop(columns=["doy_tmp"])
    return out


def build_frame(target: str) -> pd.DataFrame:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    all_dates = pd.date_range(sales["Date"].min(), FORECAST_END, freq="D")

    exog = build_exogenous(all_dates)
    exog = _fill_exog_with_doy_history(exog)

    df = pd.DataFrame({"Date": all_dates}).merge(
        sales[["Date", target]], on="Date", how="left"
    )
    df = df.merge(exog, on="Date", how="left")

    df = add_lag_rolling(df, target)
    df = add_calendar(df)

    # add lags/rolling of a handful of strong exogenous signals
    for col in [
        "orders_count",
        "orders_unique_customers",
        "items_gross_value",
        "pay_total_value",
        "web_sessions",
        "promo_active",
    ]:
        for L in (7, 14, 28):
            df[f"{col}_lag{L}"] = df[col].shift(L)
        df[f"{col}_rmean28"] = df[col].shift(1).rolling(28).mean()

    return df


EXCLUDE = {"Date"}

# Ablation/compliance guard: these same-day operational aggregates are target
# proxies on the train period (for example, items_gross_value == Revenue).
# Drop both the raw columns and their short-lag/rolling derivatives from v1.
TARGET_PROXY_BASES = {
    "orders_count",
    "items_total_qty",
    "items_gross_value",
    "pay_total_value",
    "pay_mean_value",
}


def _is_target_proxy_feature(col: str) -> bool:
    return any(base in col for base in TARGET_PROXY_BASES)


def feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    return [
        c
        for c in df.columns
        if (
            c not in EXCLUDE
            and c != target
            and df[c].dtype != "datetime64[ns]"
            and not _is_target_proxy_feature(c)
        )
    ]


def lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 4,
        "lambda_l2": 0.5,
        "verbose": -1,
        "seed": seed,
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


def recursive_forecast(df: pd.DataFrame, target: str, model: lgb.Booster, feats: list[str]) -> np.ndarray:
    """Walk forward: fill predicted target so lag/rolling can be recomputed."""
    df = df.copy().sort_values("Date").reset_index(drop=True)
    forecast_idx = df.index[df[target].isna()]
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
        yhat = float(model.predict(row[feats])[0])
        yhat = max(0.0, yhat)
        df.loc[i, target] = yhat
    return df.loc[forecast_idx, target].to_numpy()


def fit_train_yearly_trend(sales: pd.DataFrame, target: str) -> dict:
    """Log-linear yearly mean on 2015-2022 (full years, post-ramp)."""
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
        # Simple YoY continuation from 2021->2022
        "yoy_2022": float(yr.loc[2022] / yr.loc[2021]),
    }


def pick_level_target(trend: dict, target: str) -> dict:
    """Decide the per-year mean we want our forecast to hit.

    Strategy: blend log-linear trend with 2022-YoY continuation. This is
    conservative but fully train-derived.
    """
    m22 = trend["mean_2022_actual"]
    yoy = trend["yoy_2022"]
    ll_23 = trend["mean_2023_loglinear"]
    ll_24 = trend["mean_2024_loglinear"]
    # Blend 50/50 of log-linear vs. 2022*YoY
    cont_23 = m22 * yoy
    cont_24 = m22 * yoy * yoy
    t23 = 0.5 * ll_23 + 0.5 * cont_23
    t24 = 0.5 * ll_24 + 0.5 * cont_24
    return {2023: t23, 2024: t24}


def calibrate_level(pred: np.ndarray, dates, target_levels: dict) -> np.ndarray:
    out = pred.astype(float).copy()
    years = pd.DatetimeIndex(dates).year.to_numpy()
    for y, want in target_levels.items():
        mask = years == y
        have = out[mask].mean()
        if have > 0:
            out[mask] *= want / have
    return out


def run(seed: int = 42) -> dict:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    submission = pd.DataFrame({"Date": forecast_dates})

    results: dict = {"seed": seed, "metrics": [], "levels": {}, "feature_importance": {}}
    for target in ["Revenue", "COGS"]:
        df = build_frame(target)
        feats = feature_cols(df, target)

        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < "2022-01-01"]
        val = hist[hist["Date"] >= "2022-01-01"]

        dtrain = lgb.Dataset(train[feats], label=train[target])
        dval = lgb.Dataset(val[feats], label=val[target], reference=dtrain)
        model = lgb.train(
            lgb_params(seed),
            dtrain,
            num_boost_round=6000,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)],
        )

        results["metrics"].append(
            asdict(FoldMetrics("train", target, **metrics(train[target].to_numpy(), model.predict(train[feats], num_iteration=model.best_iteration))))
        )
        results["metrics"].append(
            asdict(FoldMetrics("val", target, **metrics(val[target].to_numpy(), model.predict(val[feats], num_iteration=model.best_iteration))))
        )

        # refit on full known history 2014..2022
        dfull = lgb.Dataset(hist[feats], label=hist[target])
        full_model = lgb.train(
            lgb_params(seed),
            dfull,
            num_boost_round=model.best_iteration or 2000,
            valid_sets=[dfull],
            valid_names=["train_full"],
            callbacks=[lgb.log_evaluation(0)],
        )
        # recursive forecast
        raw = recursive_forecast(df, target, full_model, feats)
        # calibration from sales-only yearly trend
        trend = fit_train_yearly_trend(sales, target)
        level_target = pick_level_target(trend, target)
        calibrated = calibrate_level(raw, forecast_dates, level_target)
        results["levels"][target] = {
            "trend_fit": trend,
            "target_levels": {str(k): v for k, v in level_target.items()},
            "raw_means": {str(y): float(raw[forecast_dates.year == y].mean()) for y in (2023, 2024)},
            "final_means": {str(y): float(calibrated[forecast_dates.year == y].mean()) for y in (2023, 2024)},
        }

        submission[target] = calibrated

        # feature importance
        gain = full_model.feature_importance(importance_type="gain")
        imp = pd.DataFrame({"feature": feats, "gain": gain}).sort_values("gain", ascending=False)
        imp.head(50).to_csv(OUT / f"feature_importance_{target.lower()}.csv", index=False)
        results["feature_importance"][target] = imp.head(20).to_dict(orient="records")
        full_model.save_model(str(OUT / f"lgbm_{target.lower()}.txt"))

    submission["Revenue"] = submission["Revenue"].round(2)
    submission["COGS"] = submission["COGS"].round(2)
    assert len(submission) == 548
    assert (submission[["Revenue", "COGS"]] > 0).all().all()
    out_path = OUT / "model_submission.csv"
    csv = submission.copy()
    csv["Date"] = csv["Date"].dt.strftime("%Y-%m-%d")
    csv.to_csv(out_path, index=False)
    results["submission_file"] = str(out_path)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = run()
    print(json.dumps(r["metrics"], indent=2))
    print("\nLevels:")
    print(json.dumps(r["levels"], indent=2))
    print(f"\nSubmission: {r['submission_file']}")
    print(f"Elapsed {time.time()-t0:.1f}s")
