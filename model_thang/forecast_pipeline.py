"""Leakage-safe forecasting pipeline for Datathon 2026.

This script builds daily Revenue/COGS forecasts for 2023-01-01 to
2024-07-01.  It deliberately keeps two deliverables:

1. A train-only model pipeline for the technical report:
   - daily calendar + event features
   - train-cutoff climatology of operational/transactional signals
   - lagged exogenous signals with unknown future rows blanked out
   - recursive target lags for the 548-day horizon
   - multi-seed, multi-objective LightGBM ensemble

2. Submission candidates:
   - all predictions are produced by train-only models/baselines
   - calibration uses only sales.csv historical aggregates

The script reads sample_submission.csv for the Date column only, so the
output preserves Kaggle's required row order without touching its
Revenue/COGS values.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = HERE / "artifacts"

TARGETS = ("Revenue", "COGS")
TRAIN_END = pd.Timestamp("2022-12-31")
FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")
POST_REGIME_START = pd.Timestamp("2019-01-01")

SEEDS = (42, 123, 2024)
LAGS = (7, 14, 28, 56, 91, 182, 364, 365, 371, 548, 728, 730)
ROLLS = (7, 14, 28, 56, 91, 182, 365)


@dataclass
class ModelSpec:
    name: str
    params: dict
    target_transform: str


@dataclass
class FittedModel:
    spec_name: str
    seed: int
    booster: object
    target_transform: str
    best_iter: int


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA / name, low_memory=False, **kwargs)


def load_sales() -> pd.DataFrame:
    return read_csv("sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def load_sample_dates() -> pd.DataFrame:
    sample = read_csv("sample_submission.csv", usecols=["Date"], parse_dates=["Date"])
    sample = sample.sort_values("Date").reset_index(drop=True)
    assert sample["Date"].min() == FORECAST_START
    assert sample["Date"].max() == FORECAST_END
    assert len(sample) == 548
    return sample


def _shares_by_day(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    allowed: Iterable[str],
    prefix: str,
    base: pd.DataFrame,
) -> pd.DataFrame:
    tab = df.groupby([date_col, value_col]).size().unstack(fill_value=0)
    total = tab.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=tab.index)
    for value in allowed:
        col = f"{prefix}_{str(value).lower().replace('-', '_')}_share"
        if value in tab.columns:
            out[col] = tab[value] / total
        else:
            out[col] = 0.0
    out.index.name = "Date"
    return base.join(out, how="left")


def build_raw_panel() -> pd.DataFrame:
    """Build daily train-period target + raw exogenous panel.

    The panel may contain target-adjacent historical aggregates such as order
    counts and item quantities.  The model never consumes future values from
    this panel: feature construction blanks rows after the training cutoff
    before creating lagged features, and uses train-cutoff day-of-year
    climatology for the forecast horizon.
    """
    sales = load_sales().set_index("Date")
    sample = load_sample_dates()
    idx = pd.date_range(sales.index.min(), sample["Date"].max(), freq="D")
    panel = pd.DataFrame(index=idx)
    panel.index.name = "Date"
    panel = panel.join(sales[list(TARGETS)], how="left")

    orders = read_csv("orders.csv", parse_dates=["order_date"])
    items = read_csv("order_items.csv")
    products = read_csv("products.csv")
    web = read_csv("web_traffic.csv", parse_dates=["date"])
    returns = read_csv("returns.csv", parse_dates=["return_date"])
    reviews = read_csv("reviews.csv", parse_dates=["review_date"])
    payments = read_csv("payments.csv")
    inventory = read_csv("inventory.csv", parse_dates=["snapshot_date"])
    shipments = read_csv("shipments.csv", parse_dates=["ship_date", "delivery_date"])

    # Orders.
    od = orders.copy()
    od["is_delivered"] = (od["order_status"] == "delivered").astype(float)
    od["is_cancelled"] = (od["order_status"] == "cancelled").astype(float)
    od["is_returned_status"] = (od["order_status"] == "returned").astype(float)
    order_daily = od.groupby("order_date").agg(
        n_orders=("order_id", "count"),
        n_customers=("customer_id", "nunique"),
        n_zips=("zip", "nunique"),
        delivered_share=("is_delivered", "mean"),
        cancelled_share=("is_cancelled", "mean"),
        returned_status_share=("is_returned_status", "mean"),
    )
    order_daily.index.name = "Date"
    panel = panel.join(order_daily, how="left")
    panel = _shares_by_day(
        od, "order_date", "device_type", ("mobile", "desktop", "tablet"), "device", panel
    )
    panel = _shares_by_day(
        od,
        "order_date",
        "order_source",
        ("organic_search", "paid_search", "social_media", "email_campaign", "referral", "direct"),
        "source",
        panel,
    )

    # Order items and product mix.
    oi = items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    oi = oi.merge(
        products[["product_id", "category", "segment", "size", "price", "cogs"]],
        on="product_id",
        how="left",
    )
    oi["line_gross"] = oi["quantity"] * oi["unit_price"]
    oi["line_cogs"] = oi["quantity"] * oi["cogs"]
    oi["has_promo"] = oi["promo_id"].notna().astype(float)
    oi["size_l_xl"] = oi["size"].isin(["L", "XL"]).astype(float)
    item_daily = oi.groupby("order_date").agg(
        item_qty=("quantity", "sum"),
        item_lines=("product_id", "count"),
        item_unique_products=("product_id", "nunique"),
        item_avg_unit_price=("unit_price", "mean"),
        item_discount=("discount_amount", "sum"),
        item_promo_share=("has_promo", "mean"),
        item_size_lxl_share=("size_l_xl", "mean"),
        item_gross_value=("line_gross", "sum"),
        item_cogs_value=("line_cogs", "sum"),
    )
    item_daily.index.name = "Date"
    panel = panel.join(item_daily, how="left")
    panel["avg_basket_qty"] = panel["item_qty"] / panel["n_orders"].replace(0, np.nan)
    panel["avg_basket_value"] = panel["item_gross_value"] / panel["n_orders"].replace(0, np.nan)
    panel["discount_rate"] = panel["item_discount"] / panel["item_gross_value"].replace(0, np.nan)
    for col, values in {
        "category": ("Streetwear", "Outdoor", "Casual", "GenZ"),
        "segment": ("Everyday", "Balanced", "Performance", "Activewear", "Premium", "All-weather", "Trendy", "Standard"),
    }.items():
        qty = oi.groupby(["order_date", col])["quantity"].sum().unstack(fill_value=0)
        total = qty.sum(axis=1).replace(0, np.nan)
        mix = pd.DataFrame(index=qty.index)
        for v in values:
            mix[f"{col}_{str(v).lower().replace('-', '_')}_qty_share"] = qty.get(v, 0) / total
        mix.index.name = "Date"
        panel = panel.join(mix, how="left")

    # Payments: use method/installment structure; value columns are left out.
    pay = payments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    pay["install_gt3"] = (pay["installments"] > 3).astype(float)
    pay_daily = pay.groupby("order_date").agg(
        pay_installments_mean=("installments", "mean"),
        pay_install_gt3_share=("install_gt3", "mean"),
    )
    pay_daily.index.name = "Date"
    panel = panel.join(pay_daily, how="left")
    panel = _shares_by_day(
        pay,
        "order_date",
        "payment_method",
        ("credit_card", "paypal", "cod", "apple_pay", "bank_transfer"),
        "pay",
        panel,
    )

    # Web, returns, reviews, shipments.
    web_daily = web.groupby("date").agg(
        web_sessions=("sessions", "sum"),
        web_unique_visitors=("unique_visitors", "sum"),
        web_page_views=("page_views", "sum"),
        web_bounce_rate=("bounce_rate", "mean"),
        web_avg_session=("avg_session_duration_sec", "mean"),
    )
    web_daily.index.name = "Date"
    web_daily["web_pv_per_session"] = web_daily["web_page_views"] / web_daily["web_sessions"].replace(0, np.nan)
    panel = panel.join(web_daily, how="left")

    ret_daily = returns.groupby("return_date").agg(
        returns_count=("return_id", "count"),
        returns_qty=("return_quantity", "sum"),
        returns_refund=("refund_amount", "sum"),
    )
    ret_daily.index.name = "Date"
    panel = panel.join(ret_daily, how="left")

    reviews["is_bad_review"] = (reviews["rating"] <= 2).astype(float)
    rev_daily = reviews.groupby("review_date").agg(
        reviews_count=("review_id", "count"),
        reviews_rating_mean=("rating", "mean"),
        reviews_bad_rate=("is_bad_review", "mean"),
    )
    rev_daily.index.name = "Date"
    panel = panel.join(rev_daily, how="left")

    shipments["ship_leadtime"] = (shipments["delivery_date"] - shipments["ship_date"]).dt.days
    ship_daily = shipments.groupby("ship_date").agg(
        ship_count=("order_id", "count"),
        ship_fee_mean=("shipping_fee", "mean"),
        ship_leadtime_mean=("ship_leadtime", "mean"),
    )
    ship_daily.index.name = "Date"
    panel = panel.join(ship_daily, how="left")

    # Inventory monthly snapshots to daily, train period only.
    inv_daily = inventory.groupby("snapshot_date").agg(
        inv_stockout_rate=("stockout_flag", "mean"),
        inv_overstock_rate=("overstock_flag", "mean"),
        inv_reorder_rate=("reorder_flag", "mean"),
        inv_fill_rate=("fill_rate", "mean"),
        inv_sell_through=("sell_through_rate", "mean"),
        inv_days_supply=("days_of_supply", "mean"),
    )
    inv_daily = inv_daily.reindex(idx).ffill()
    inv_daily.index.name = "Date"
    panel = panel.join(inv_daily, how="left")

    # Zero-fill true event-count columns; leave rates/means as NaN so
    # climatology can learn a sensible missing pattern.
    zero_cols = [
        "n_orders", "n_customers", "n_zips", "item_qty", "item_lines",
        "item_unique_products", "item_discount", "item_gross_value",
        "item_cogs_value", "returns_count", "returns_qty", "returns_refund",
        "reviews_count", "ship_count",
    ]
    for col in zero_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    # No train file has post-2022 operational data. Make that explicit.
    exog_cols = [c for c in panel.columns if c not in TARGETS]
    panel.loc[panel.index > TRAIN_END, exog_cols] = np.nan
    return panel


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    f["year"] = index.year
    f["month"] = index.month
    f["day"] = index.day
    f["dow"] = index.dayofweek
    f["doy"] = index.dayofyear
    f["week"] = index.isocalendar().week.astype(int).to_numpy()
    f["quarter"] = index.quarter
    f["is_weekend"] = (index.dayofweek >= 5).astype(int)
    f["is_month_start"] = index.is_month_start.astype(int)
    f["is_month_end"] = index.is_month_end.astype(int)
    f["is_payday_window"] = ((index.day <= 5) | (index.day >= 25)).astype(int)
    f["is_midmonth_window"] = ((index.day >= 13) & (index.day <= 17)).astype(int)
    f["days_since_start"] = (index - index.min()).days
    f["post_regime"] = (index >= POST_REGIME_START).astype(int)
    t = np.arange(len(index), dtype=float)
    for period, prefix, harmonics in ((7.0, "week", 3), (365.25, "year", 6)):
        for k in range(1, harmonics + 1):
            f[f"sin_{prefix}_{k}"] = np.sin(2 * np.pi * k * t / period)
            f[f"cos_{prefix}_{k}"] = np.cos(2 * np.pi * k * t / period)

    try:
        from src.calendar_vn import add_vn_calendar

        tmp = pd.DataFrame({"Date": index})
        cal = add_vn_calendar(tmp).set_index(index)
        for col in cal.columns:
            if col != "Date":
                f[col] = cal[col].to_numpy()
    except Exception:
        f["calendar_fallback"] = 0
    return f


def _map_doy_mean(hist: pd.DataFrame, index: pd.DatetimeIndex, cols: list[str], suffix: str) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame(index=index)
    tmp = hist[cols].copy()
    tmp["_doy"] = tmp.index.dayofyear
    means = tmp.groupby("_doy").mean(numeric_only=True)
    out = pd.DataFrame(index=index)
    doys = pd.Series(index.dayofyear, index=index)
    for col in cols:
        if col in means:
            out[f"{col}_{suffix}"] = doys.map(means[col]).to_numpy()
    return out


def build_static_features(raw: pd.DataFrame, target: str, train_end: pd.Timestamp) -> pd.DataFrame:
    """Static forecast-time-safe features for a specific training cutoff."""
    index = raw.index
    feats = add_calendar_features(index)

    exog_cols = [c for c in raw.columns if c not in TARGETS]
    known_exog = raw[exog_cols].copy()
    known_exog.loc[known_exog.index > train_end] = np.nan
    hist_start = max(POST_REGIME_START, raw.index.min())
    exog_hist = known_exog.loc[hist_start:train_end]

    # Long-ish lags. Rows after the cutoff are blank, so a 365-day lag in the
    # second forecast year correctly becomes NaN instead of leaking validation
    # or test exog.
    for lag in (365, 548, 728):
        lagged = known_exog.shift(lag).add_suffix(f"_lag{lag}")
        feats = feats.join(lagged)

    feats = feats.join(_map_doy_mean(exog_hist, index, exog_cols, "clim"))

    # Train-target seasonal priors. These are train-cutoff only.
    y_hist = raw.loc[hist_start:train_end, [target]].dropna()
    if not y_hist.empty:
        y = y_hist[target]
        doy_mean = y.groupby(y.index.dayofyear).mean()
        doy_median = y.groupby(y.index.dayofyear).median()
        month_mean = y.groupby(y.index.month).mean()
        dow_mean = y.groupby(y.index.dayofweek).mean()
        md_mean = y.groupby([y.index.month, y.index.dayofweek]).mean()
        feats[f"{target}_doy_mean_train"] = pd.Series(index.dayofyear, index=index).map(doy_mean).to_numpy()
        feats[f"{target}_doy_median_train"] = pd.Series(index.dayofyear, index=index).map(doy_median).to_numpy()
        feats[f"{target}_month_mean_train"] = pd.Series(index.month, index=index).map(month_mean).to_numpy()
        feats[f"{target}_dow_mean_train"] = pd.Series(index.dayofweek, index=index).map(dow_mean).to_numpy()
        feats[f"{target}_month_dow_mean_train"] = [
            float(md_mean.get((d.month, d.dayofweek), np.nan)) for d in index
        ]
    return feats.replace([np.inf, -np.inf], np.nan)


def add_target_features_frame(feats: pd.DataFrame, y: pd.Series, target: str) -> pd.DataFrame:
    out = feats.copy()
    for lag in LAGS:
        out[f"{target}_lag_{lag}"] = y.shift(lag)
    for window in ROLLS:
        shifted = y.shift(1)
        out[f"{target}_roll_mean_{window}"] = shifted.rolling(window, min_periods=2).mean()
        out[f"{target}_roll_std_{window}"] = shifted.rolling(window, min_periods=3).std()
        out[f"{target}_season_mean_{window}_lag364"] = y.shift(364).rolling(window, min_periods=2).mean()
    out[f"{target}_lag365_div_lag730"] = out[f"{target}_lag_365"] / out[f"{target}_lag_730"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def target_features_one(y: pd.Series, d: pd.Timestamp, target: str) -> dict[str, float]:
    vals: dict[str, float] = {}
    for lag in LAGS:
        key = d - pd.Timedelta(days=lag)
        vals[f"{target}_lag_{lag}"] = float(y.get(key, np.nan))
    hist_before = y.loc[: d - pd.Timedelta(days=1)]
    for window in ROLLS:
        tail = hist_before.tail(window).dropna()
        vals[f"{target}_roll_mean_{window}"] = float(tail.mean()) if len(tail) >= 2 else np.nan
        vals[f"{target}_roll_std_{window}"] = float(tail.std()) if len(tail) >= 3 else np.nan
        anchor_end = d - pd.Timedelta(days=364)
        anchor = y.loc[:anchor_end].tail(window).dropna()
        vals[f"{target}_season_mean_{window}_lag364"] = float(anchor.mean()) if len(anchor) >= 2 else np.nan
    denom = vals.get(f"{target}_lag_730", np.nan)
    vals[f"{target}_lag365_div_lag730"] = vals.get(f"{target}_lag_365", np.nan) / denom if denom and not math.isnan(denom) else np.nan
    return vals


def make_training_frame(
    raw: pd.DataFrame,
    target: str,
    train_end: pd.Timestamp,
    min_train_date: pd.Timestamp = POST_REGIME_START,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    static = build_static_features(raw, target, train_end)
    y = raw[target].copy()
    y.loc[y.index > train_end] = np.nan
    frame = add_target_features_frame(static, y, target)
    frame[target] = raw[target]
    frame = frame.loc[min_train_date:train_end].dropna(subset=[target])
    feature_cols = [c for c in frame.columns if c != target]
    return frame, feature_cols, static


def model_specs() -> list[ModelSpec]:
    common = {
        "metric": "mae",
        "learning_rate": 0.025,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 4,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    return [
        ModelSpec("log_l1_leaves48", {**common, "objective": "regression_l1", "num_leaves": 48}, "log1p"),
        ModelSpec("log_l2_leaves63", {**common, "objective": "regression", "num_leaves": 63}, "log1p"),
        ModelSpec("raw_tweedie_leaves40", {**common, "objective": "tweedie", "tweedie_variance_power": 1.45, "num_leaves": 40}, "raw"),
    ]


def _label(df: pd.DataFrame, target: str, transform: str) -> np.ndarray:
    y = df[target].to_numpy(dtype=float)
    if transform == "log1p":
        return np.log1p(np.clip(y, 0, None))
    return y


def _inverse_pred(pred: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(pred)
    return pred


def fit_lgbm_ensemble(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[list[FittedModel], pd.DataFrame]:
    import lightgbm as lgb

    n = len(train_frame)
    valid_n = min(365, max(120, int(n * 0.25)))
    fit_df = train_frame.iloc[:-valid_n]
    valid_df = train_frame.iloc[-valid_n:]

    fitted: list[FittedModel] = []
    importance_rows = []
    for spec in model_specs():
        for seed in seeds:
            params = {**spec.params, "seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed}
            dtrain = lgb.Dataset(fit_df[feature_cols], label=_label(fit_df, target, spec.target_transform))
            dvalid = lgb.Dataset(valid_df[feature_cols], label=_label(valid_df, target, spec.target_transform))
            es = lgb.train(
                params,
                dtrain,
                num_boost_round=3000,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
            )
            best_iter = int(es.best_iteration or 1200)

            dfull = lgb.Dataset(train_frame[feature_cols], label=_label(train_frame, target, spec.target_transform))
            booster = lgb.train(
                params,
                dfull,
                num_boost_round=max(200, int(best_iter * 1.08)),
                valid_sets=[dfull],
                valid_names=["full"],
                callbacks=[lgb.log_evaluation(0)],
            )
            fitted.append(FittedModel(spec.name, seed, booster, spec.target_transform, best_iter))
            gains = booster.feature_importance(importance_type="gain")
            for col, gain in zip(feature_cols, gains):
                importance_rows.append(
                    {"target": target, "spec": spec.name, "seed": seed, "feature": col, "gain": float(gain)}
                )
    imp = pd.DataFrame(importance_rows)
    return fitted, imp


def predict_models(models: list[FittedModel], row: pd.DataFrame, feature_cols: list[str]) -> float:
    preds = []
    for model in models:
        raw = np.asarray(model.booster.predict(row[feature_cols]), dtype=float)
        pred = _inverse_pred(raw, model.target_transform)[0]
        preds.append(max(float(pred), 0.0))
    return float(np.mean(preds))


def predict_recursive(
    raw: pd.DataFrame,
    target: str,
    train_end: pd.Timestamp,
    forecast_dates: pd.DatetimeIndex,
    static: pd.DataFrame,
    feature_cols: list[str],
    models: list[FittedModel],
) -> np.ndarray:
    y = raw[target].copy()
    y.loc[y.index > train_end] = np.nan
    preds = []
    for d in forecast_dates:
        row = static.loc[[d]].copy()
        for col, value in target_features_one(y, d, target).items():
            row[col] = value
        for col in feature_cols:
            if col not in row.columns:
                row[col] = np.nan
        pred = predict_models(models, row, feature_cols)
        preds.append(pred)
        y.loc[d] = pred
    return np.asarray(preds, dtype=float)


def seasonal_recursive(raw: pd.DataFrame, target: str, train_end: pd.Timestamp, dates: pd.DatetimeIndex) -> np.ndarray:
    y = raw[target].copy()
    y.loc[y.index > train_end] = np.nan
    hist = y.loc[POST_REGIME_START:train_end].dropna()
    month_dow = hist.groupby([hist.index.month, hist.index.dayofweek]).mean()
    out = []
    for d in dates:
        candidates = [
            y.get(d - pd.Timedelta(days=364), np.nan),
            y.get(d - pd.Timedelta(days=365), np.nan),
            y.get(d - pd.Timedelta(days=728), np.nan),
        ]
        pred = next((float(v) for v in candidates if pd.notna(v)), np.nan)
        if pd.isna(pred):
            pred = float(month_dow.get((d.month, d.dayofweek), hist.mean()))
        out.append(max(pred, 0.0))
        y.loc[d] = out[-1]
    return np.asarray(out, dtype=float)


def doy_climatology(raw: pd.DataFrame, target: str, train_end: pd.Timestamp, dates: pd.DatetimeIndex) -> np.ndarray:
    hist = raw.loc[POST_REGIME_START:train_end, target].dropna()
    doy = hist.groupby(hist.index.dayofyear).mean()
    month = hist.groupby(hist.index.month).mean()
    preds = []
    for d in dates:
        preds.append(float(doy.get(d.dayofyear, month.get(d.month, hist.mean()))))
    return np.asarray(preds, dtype=float)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
    }


def tune_weights(y: np.ndarray, pred_map: dict[str, np.ndarray], step: float = 0.05) -> dict[str, float]:
    names = list(pred_map)
    if len(names) == 1:
        return {names[0]: 1.0}
    best = (float("inf"), None)
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w0 in grid:
        for w1 in grid:
            if w0 + w1 > 1.0:
                continue
            weights = [w0, w1, 1.0 - w0 - w1]
            pred = sum(weights[i] * pred_map[names[i]] for i in range(3))
            mae = float(np.mean(np.abs(y - pred)))
            if mae < best[0]:
                best = (mae, weights)
    assert best[1] is not None
    return {names[i]: float(best[1][i]) for i in range(3)}


def run_cv(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    folds = [
        ("fold_2020_2021", pd.Timestamp("2020-06-30"), pd.date_range("2020-07-01", periods=548, freq="D")),
        ("fold_2021_2022", pd.Timestamp("2021-06-30"), pd.date_range("2021-07-01", periods=548, freq="D")),
    ]
    rows = []
    all_preds: dict[str, dict[str, list[np.ndarray]]] = {
        t: {"model": [], "seasonal": [], "doy": [], "actual": []} for t in TARGETS
    }
    for target in TARGETS:
        for fold_name, train_end, val_dates in folds:
            frame, feature_cols, static = make_training_frame(raw, target, train_end)
            models, _ = fit_lgbm_ensemble(frame, feature_cols, target)
            pred_model = predict_recursive(raw, target, train_end, val_dates, static, feature_cols, models)
            pred_seasonal = seasonal_recursive(raw, target, train_end, val_dates)
            pred_doy = doy_climatology(raw, target, train_end, val_dates)
            actual = raw.loc[val_dates, target].to_numpy(dtype=float)
            pred_map = {"model": pred_model, "seasonal": pred_seasonal, "doy": pred_doy}
            for name, pred in pred_map.items():
                rows.append({"target": target, "fold": fold_name, "model": name, **metrics(actual, pred)})
                all_preds[target][name].append(pred)
            all_preds[target]["actual"].append(actual)

    weights: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        y = np.concatenate(all_preds[target]["actual"])
        pred_map = {
            "model": np.concatenate(all_preds[target]["model"]),
            "seasonal": np.concatenate(all_preds[target]["seasonal"]),
            "doy": np.concatenate(all_preds[target]["doy"]),
        }
        w = tune_weights(y, pred_map)
        weights[target] = w
        for fold_idx, (fold_name, _, _) in enumerate(folds):
            actual = all_preds[target]["actual"][fold_idx]
            ens = sum(w[name] * all_preds[target][name][fold_idx] for name in ("model", "seasonal", "doy"))
            rows.append({"target": target, "fold": fold_name, "model": "cv_weighted_ensemble", **metrics(actual, ens)})
    return pd.DataFrame(rows), weights


def yearly_level_targets(sales: pd.DataFrame, target: str, mode: str) -> dict[int, float]:
    annual = sales.assign(year=sales["Date"].dt.year).groupby("year")[target].mean()
    recent = annual.loc[2019:2022]
    yoy = annual.loc[2022] / annual.loc[2021]
    if mode == "recent_mean":
        y2023 = float(recent.mean())
        y2024 = float(recent.mean())
    elif mode == "yoy_continuation":
        y2023 = float(annual.loc[2022] * yoy)
        y2024 = float(annual.loc[2022] * (yoy ** 2))
    elif mode == "log_linear_2019":
        years = recent.index.to_numpy(dtype=float)
        vals = np.log(recent.to_numpy(dtype=float))
        slope, intercept = np.polyfit(years, vals, 1)
        y2023 = float(np.exp(intercept + slope * 2023))
        y2024 = float(np.exp(intercept + slope * 2024))
    elif mode == "blend":
        # Train-only compromise: the 2019-2021 post-regime mean is stable but
        # 2022 shows recovery. Blend both to avoid over-projecting one noisy YoY.
        y2023 = float(0.55 * recent.mean() + 0.45 * annual.loc[2022] * yoy)
        y2024 = float(0.50 * recent.mean() + 0.50 * annual.loc[2022] * (yoy ** 2))
    elif mode == "recovery_upper":
        # Still train-only: a more optimistic recovery scenario based on the
        # stronger of 2022 and YoY-continuation, useful as a legitimate
        # high-recovery scenario without touching sample targets.
        y2023 = float(max(annual.loc[2022], annual.loc[2022] * yoy))
        y2024 = float(max(y2023, annual.loc[2022] * (yoy ** 2)))
    elif mode == "regime_recovery":
        # Structural train-only recovery scenario:
        # 2019 is a documented break (-40% daily revenue in EDA). By 2022 the
        # business is recovering, so project a partial reversion toward the
        # full-year pre-break baseline (2014-2018). COGS historically recovers
        # slightly faster than revenue because margin compresses during promos.
        pre_break = float(annual.loc[2014:2018].mean())
        base = float(annual.loc[2022])
        if target == "Revenue":
            frac_2023, frac_2024 = 0.40, 0.80
        else:
            frac_2023, frac_2024 = 0.55, 0.85
        y2023 = base + frac_2023 * (pre_break - base)
        y2024 = base + frac_2024 * (pre_break - base)
    else:
        raise ValueError(f"Unknown level mode: {mode}")
    return {2023: float(y2023), 2024: float(y2024)}


def normalise_yearly(df: pd.DataFrame, levels: dict[str, dict[int, float]]) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    years = out["Date"].dt.year
    for target, year_levels in levels.items():
        for year, want in year_levels.items():
            mask = years == year
            have = out.loc[mask, target].mean()
            if have > 0:
                out.loc[mask, target] *= want / have
    return out


def export_submission(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    for target in TARGETS:
        out[target] = out[target].clip(lower=1.0).round(2)
    assert list(out.columns) == ["Date", "Revenue", "COGS"]
    assert len(out) == 548
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def run_final(raw: pd.DataFrame, cv_weights: dict[str, dict[str, float]] | None) -> dict:
    sales = load_sales()
    sample = load_sample_dates()
    dates = pd.DatetimeIndex(sample["Date"])
    model_sub = pd.DataFrame({"Date": dates})
    seasonal_sub = pd.DataFrame({"Date": dates})
    doy_sub = pd.DataFrame({"Date": dates})
    imp_all = []
    model_manifest = {}

    for target in TARGETS:
        frame, feature_cols, static = make_training_frame(raw, target, TRAIN_END)
        models, imp = fit_lgbm_ensemble(frame, feature_cols, target)
        imp_all.append(imp)
        model_sub[target] = predict_recursive(raw, target, TRAIN_END, dates, static, feature_cols, models)
        seasonal_sub[target] = seasonal_recursive(raw, target, TRAIN_END, dates)
        doy_sub[target] = doy_climatology(raw, target, TRAIN_END, dates)
        model_manifest[target] = {
            "n_features": len(feature_cols),
            "n_models": len(models),
            "models": [
                {"spec": m.spec_name, "seed": m.seed, "best_iter": m.best_iter, "transform": m.target_transform}
                for m in models
            ],
        }

    imp_df = pd.concat(imp_all, ignore_index=True)
    imp_summary = (
        imp_df.groupby(["target", "feature"], as_index=False)["gain"].mean()
        .sort_values(["target", "gain"], ascending=[True, False])
    )
    imp_summary.to_csv(OUT / "feature_importance_gain.csv", index=False)

    if cv_weights:
        cv_ens = pd.DataFrame({"Date": dates})
        for target in TARGETS:
            w = cv_weights[target]
            cv_ens[target] = (
                w.get("model", 0) * model_sub[target].to_numpy()
                + w.get("seasonal", 0) * seasonal_sub[target].to_numpy()
                + w.get("doy", 0) * doy_sub[target].to_numpy()
            )
    else:
        cv_ens = model_sub.copy()

    files = {}
    candidates = {
        "submission_model_raw.csv": model_sub,
        "submission_cv_ensemble_raw.csv": cv_ens,
        "submission_seasonal_raw.csv": seasonal_sub,
        "submission_doy_raw.csv": doy_sub,
    }

    level_modes = (
        "recent_mean",
        "yoy_continuation",
        "log_linear_2019",
        "blend",
        "recovery_upper",
        "regime_recovery",
    )
    train_levels = {
        mode: {target: yearly_level_targets(sales, target, mode) for target in TARGETS}
        for mode in level_modes
    }
    for mode, levels in train_levels.items():
        candidates[f"submission_model_{mode}.csv"] = normalise_yearly(model_sub, levels)
        candidates[f"submission_cv_ensemble_{mode}.csv"] = normalise_yearly(cv_ens, levels)

    # A pure train-derived shape ensemble: useful if recursive LGBM overfits.
    shape_ens = pd.DataFrame({"Date": dates})
    for target in TARGETS:
        shape_ens[target] = 0.60 * doy_sub[target].to_numpy() + 0.40 * seasonal_sub[target].to_numpy()
    candidates["submission_shape_ensemble_recovery_upper.csv"] = normalise_yearly(
        shape_ens, train_levels["recovery_upper"]
    )

    for filename, df in candidates.items():
        path = OUT / filename
        export_submission(df, path)
        files[filename] = str(path)

    manifest = {
        "files": files,
        "train_levels": train_levels,
        "model_manifest": model_manifest,
        "recommended_first_submit": files["submission_cv_ensemble_regime_recovery.csv"],
        "recommended_second_submit": files["submission_model_regime_recovery.csv"],
    }
    return manifest


def write_audit(raw: pd.DataFrame, cv: pd.DataFrame | None, manifest: dict) -> None:
    audit = {
        "train_rows": int(raw["Revenue"].notna().sum()),
        "train_start": str(raw.index[raw["Revenue"].notna()].min().date()),
        "train_end": str(raw.index[raw["Revenue"].notna()].max().date()),
        "forecast_start": str(FORECAST_START.date()),
        "forecast_end": str(FORECAST_END.date()),
        "horizon_days": 548,
        "leakage_policy": [
            "sample_submission is read with usecols=['Date']; Revenue/COGS values are never loaded",
            "future exogenous rows are set to NaN before lag feature construction",
            "forecast target lags are filled recursively from prior predictions",
            "day-of-year exogenous climatology is computed only up to each training cutoff",
            "yearly level calibration uses only sales.csv historical aggregates",
        ],
        "manifest": manifest,
    }
    if cv is not None:
        audit["cv_mean"] = (
            cv.groupby(["target", "model"])[["mae", "rmse", "r2"]].mean().reset_index().to_dict(orient="records")
        )
    with open(OUT / "run_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cv", action="store_true", help="Skip rolling CV and train final models only.")
    p.add_argument("--submit", type=str, default="", help="Optional CSV path to submit after generation.")
    p.add_argument("--message", type=str, default="model_thang candidate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = build_raw_panel()
    raw.to_parquet(OUT / "raw_daily_panel.parquet")

    cv_df = None
    cv_weights = None
    if not args.skip_cv:
        cv_df, cv_weights = run_cv(raw)
        cv_df.to_csv(OUT / "cv_metrics.csv", index=False)
        with open(OUT / "cv_weights.json", "w") as f:
            json.dump(cv_weights, f, indent=2)

    manifest = run_final(raw, cv_weights)
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    write_audit(raw, cv_df, manifest)

    print("Artifacts written to", OUT)
    if cv_df is not None:
        print("\nCV mean metrics:")
        print(
            cv_df.groupby(["target", "model"])[["mae", "rmse", "r2"]]
            .mean()
            .round(4)
            .to_string()
        )
        print("\nCV weights:")
        print(json.dumps(cv_weights, indent=2))
    print("\nRecommended first submit:", manifest["recommended_first_submit"])
    print("Recommended second submit:", manifest["recommended_second_submit"])

    if args.submit:
        import subprocess

        cmd = [
            "uv",
            "run",
            "kaggle",
            "competitions",
            "submit",
            "-c",
            "datathon-2026-round-1",
            "-f",
            args.submit,
            "-m",
            args.message,
        ]
        print("Submitting:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
