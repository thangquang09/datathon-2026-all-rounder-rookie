"""End-to-end forecasting pipeline for Datathon 2026 — The Gridbreakers (Part 3).

Predicts daily Revenue and COGS for 2023-01-01 → 2024-07-01 (548 days).

This module is imported by modeling.ipynb. Everything is written as pure
functions so the notebook stays linear and reproducible.

Design decisions (from results/v2/report.md):
  - Insight 1: regime break in 2019 → train on 2019-01-01 onward only.
  - Insight 2: strong seasonality (May peak, Dec/Jan trough) → calendar + Fourier.
  - Insight 6: n_orders and refund (lagged) are the strongest exogenous signals.
  - Insight 9: rolling-origin CV with 548-day validation windows is mandatory.
  - Horizon 548 days is longer than any naturally-available recent lag, so
    every exogenous feature must be either (a) lagged by ≥ 548 days, or
    (b) replaced with a day-of-year climatology (long-term average) when the
    lagged value does not exist at forecast time. We choose (b) for exogenous
    features and (a) for the target itself (lag 364 / 728).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data")
OUT_DIR = Path("results/v3")

REGIME_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-07-01")

TARGETS = ("Revenue", "COGS")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_sales() -> pd.DataFrame:
    """Train target series, indexed by calendar date."""
    df = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_sample_submission() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    return df


def build_daily_panel() -> pd.DataFrame:
    """Join target + exogenous signals at daily resolution.

    All exogenous features are aggregated to the day. We DO NOT attach
    any same-day signal that would leak the target (n_orders_t, COGS_t,
    refund_t). We store raw daily aggregates; lagging is done later in
    `add_features`.

    Returns dataframe indexed by Date, with columns:
      Revenue, COGS, n_orders, avg_basket, refund_amount, sessions,
      unique_visitors, page_views, bounce_rate, promo_share, total_discount
    """
    sales = load_sales().set_index("Date")

    # Orders ------------------------------------------------------------- #
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(DATA_DIR / "order_items.csv")
    oi = order_items.merge(
        orders[["order_id", "order_date"]], on="order_id", how="left"
    )
    oi["gross"] = oi["quantity"] * oi["unit_price"]
    daily_orders = (
        oi.groupby("order_date")
        .agg(
            n_orders=("order_id", "nunique"),
            n_items=("quantity", "sum"),
            gross_rev=("gross", "sum"),
            total_discount=("discount_amount", "sum"),
            promo_orders=("promo_id", lambda s: s.notna().sum()),
        )
        .rename_axis("Date")
    )
    daily_orders["avg_basket"] = daily_orders["gross_rev"] / daily_orders[
        "n_orders"
    ].replace(0, np.nan)
    daily_orders["promo_share"] = (
        daily_orders["promo_orders"] / daily_orders["n_orders"].replace(0, np.nan)
    ).fillna(0.0)

    # Web traffic -------------------------------------------------------- #
    web = pd.read_csv(DATA_DIR / "web_traffic.csv", parse_dates=["date"])
    web_daily = web.groupby("date").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
    ).rename_axis("Date")

    # Returns ------------------------------------------------------------ #
    ret = pd.read_csv(DATA_DIR / "returns.csv", parse_dates=["return_date"])
    refund_daily = (
        ret.groupby("return_date")
        .agg(refund_amount=("refund_amount", "sum"), n_returns=("return_id", "count"))
        .rename_axis("Date")
    )

    # Inventory (monthly snapshot → forward-fill to daily) ---------------- #
    inv = pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["snapshot_date"])
    inv_monthly = inv.groupby("snapshot_date").agg(
        stockout_rate=("stockout_flag", "mean"),
        overstock_rate=("overstock_flag", "mean"),
        mean_fill_rate=("fill_rate", "mean"),
        days_of_supply=("days_of_supply", "mean"),
    ).rename_axis("Date")
    # Forward fill to daily — snapshot at month-end applies for ~30 days
    # Then lag by 30d later when building features (Insight 5 leakage rule).
    daily_index = pd.date_range(sales.index.min(), sales.index.max(), freq="D")
    inv_daily = inv_monthly.reindex(daily_index).ffill().bfill()
    inv_daily.index.name = "Date"

    # Merge -------------------------------------------------------------- #
    panel = sales.copy()
    for df in (daily_orders, web_daily, refund_daily, inv_daily):
        panel = panel.join(df, how="left")
    panel = panel.sort_index()
    # Small gaps (days with no orders or no returns) → 0
    fill0 = [
        "n_orders", "n_items", "gross_rev", "total_discount", "promo_orders",
        "promo_share", "refund_amount", "n_returns",
    ]
    for c in fill0:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)
    return panel


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Pure calendar features — all available at forecast time."""
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek
    f["day"] = idx.day
    f["month"] = idx.month
    f["quarter"] = idx.quarter
    f["week_of_year"] = idx.isocalendar().week.astype(int)
    f["day_of_year"] = idx.dayofyear
    f["is_month_start"] = idx.is_month_start.astype(int)
    f["is_month_end"] = idx.is_month_end.astype(int)
    f["is_quarter_start"] = idx.is_quarter_start.astype(int)
    f["is_quarter_end"] = idx.is_quarter_end.astype(int)
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["year"] = idx.year

    # Fourier terms (weekly + yearly). Annual captures May-peak / Dec-trough.
    t = np.arange(len(idx))
    for K, period, name in [(2, 7.0, "w"), (4, 365.25, "y")]:
        for k in range(1, K + 1):
            f[f"fourier_sin_{name}_{k}"] = np.sin(2 * np.pi * k * t / period)
            f[f"fourier_cos_{name}_{k}"] = np.cos(2 * np.pi * k * t / period)

    # Vietnamese holidays ------------------------------------------------ #
    try:
        import holidays
        vn = holidays.country_holidays("VN", years=range(idx.min().year, idx.max().year + 1))
        f["is_holiday"] = idx.to_series().isin(vn).astype(int).values
    except Exception:
        f["is_holiday"] = 0

    # Regime flag (Insight 1) ------------------------------------------- #
    f["is_post_regime"] = (idx >= REGIME_START).astype(int)

    return f


def _target_lags(y: pd.Series, lags: list[int], rolls: list[int]) -> pd.DataFrame:
    """Target autoregressive features. Lag must be ≥ 364 to be known on
    the farthest forecast horizon (test spans 548 days; lag 364 means the
    last 184 forecast days would still need lag 364 which points into the
    test window itself — we handle this by iterative filling during
    inference or by using only lag 364 which is available at t_test_start
    for all t in the first year, plus lag 728 for the second year).

    For CV we use lag 364 (available). For final inference we use a
    'static' feature set where lag_364 points into 2022 for 2023 dates
    and into 2023 (predicted) for 2024 dates → recursive.
    """
    f = pd.DataFrame(index=y.index)
    for L in lags:
        f[f"{y.name}_lag_{L}"] = y.shift(L)
    for W in rolls:
        # shifted rolling so the feature at date t uses only data ≤ t − 364.
        f[f"{y.name}_roll_mean_{W}_lag364"] = y.shift(364).rolling(W, min_periods=1).mean()
        f[f"{y.name}_roll_std_{W}_lag364"] = y.shift(364).rolling(W, min_periods=1).std()
    return f


def _exog_climatology(panel: pd.DataFrame, exog_cols: list[str]) -> pd.DataFrame:
    """For each exogenous column, compute a day-of-year climatology from
    the post-regime years (2019–2022). This gives a forecast-time-safe
    estimate of each signal for every date in the test window.

    We also return a lag-365 copy to preserve recent-year momentum.
    """
    use = panel.loc[REGIME_START:TRAIN_END, exog_cols].copy()
    use["doy"] = use.index.dayofyear
    clim = use.groupby("doy").mean(numeric_only=True)
    return clim  # index: doy(1..366), columns: exog_cols


def add_features(
    panel: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full feature matrix for `target` (Revenue or COGS).

    Returns: (df_with_features, feature_columns)
    """
    assert target in panel.columns

    idx = panel.index
    feats = _calendar_features(idx)

    # Target autoregressive --------------------------------------------- #
    y = panel[target]
    feats = feats.join(
        _target_lags(y, lags=[364, 365, 728], rolls=[7, 28, 91, 364])
    )

    # Exogenous climatology (forecast-safe) ----------------------------- #
    exog_cols = [
        "n_orders", "avg_basket", "promo_share", "total_discount",
        "sessions", "unique_visitors", "page_views", "bounce_rate",
        "refund_amount", "stockout_rate", "overstock_rate",
        "mean_fill_rate", "days_of_supply",
    ]
    exog_cols = [c for c in exog_cols if c in panel.columns]

    # Lag-365 of exogenous — known for 2023 dates (points into 2022)
    lag365 = panel[exog_cols].shift(365).add_suffix("_lag365")
    feats = feats.join(lag365)

    # Long lag (≥548) — strictly forecast-safe for ALL test dates
    lag548 = panel[exog_cols].shift(548).add_suffix("_lag548")
    feats = feats.join(lag548)

    # Climatology from post-regime years (2019-2022) — static per doy
    clim = _exog_climatology(panel, exog_cols)
    clim_df = pd.DataFrame(index=idx)
    clim_df["doy"] = idx.dayofyear
    clim_df = clim_df.merge(
        clim.add_suffix("_clim"), left_on="doy", right_index=True, how="left"
    ).drop(columns="doy")
    clim_df.index = idx
    feats = feats.join(clim_df)

    # The target column itself (for fitting); keep as separate column.
    feats[target] = y.values

    feature_cols = [c for c in feats.columns if c != target]
    return feats, feature_cols


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    name: str
    y_pred: np.ndarray
    metrics: dict[str, float]


def seasonal_naive(
    full_target: pd.Series, val_index: pd.DatetimeIndex
) -> np.ndarray:
    """Seasonal-naive: Rev(t) = Rev(t − 364). The value is sourced from
    full_target (must contain train). Falls back to lag 365 then to the
    rolling mean of the same (month, dow) if lag 364 is missing.
    """
    shifted_364 = full_target.shift(364)
    shifted_365 = full_target.shift(365)
    pred = shifted_364.reindex(val_index)
    pred = pred.fillna(shifted_365.reindex(val_index))

    # Fallback: (month, dow) mean over post-regime years
    missing = pred.isna()
    if missing.any():
        post = full_target.loc[REGIME_START:]
        key = pd.DataFrame(
            {"m": post.index.month, "d": post.index.dayofweek, "y": post.values},
        )
        group_mean = key.groupby(["m", "d"])["y"].mean()
        fill = pd.Series(
            [
                group_mean.get((ts.month, ts.dayofweek), full_target.mean())
                for ts in val_index
            ],
            index=val_index,
        )
        pred = pred.fillna(fill)
    return pred.values


def fit_sarimax(
    train_y: pd.Series,
    val_index: pd.DatetimeIndex,
    full_y: pd.Series,
) -> np.ndarray:
    """SARIMAX(1,1,1)(1,1,1)_7 on log-revenue with Fourier(365.25) exog.

    Fourier origin is anchored to train_start so that exog values at
    validation time are not extrapolated to unfamiliar magnitudes.
    Predictions are clipped to [0, 3 × max(train)] to guard against
    non-stationary blow-ups on this volatile regime.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    log_y = np.log1p(train_y.values)
    origin = train_y.index.min()

    def fourier_exog(idx: pd.DatetimeIndex) -> np.ndarray:
        t = (idx - origin).days.values.astype(float)
        cols = []
        # Weekly Fourier (period 7)
        for k in range(1, 3):
            cols.append(np.sin(2 * np.pi * k * t / 7.0))
            cols.append(np.cos(2 * np.pi * k * t / 7.0))
        # Yearly Fourier (period 365.25)
        for k in range(1, 5):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))
        return np.column_stack(cols)

    exog_train = fourier_exog(train_y.index)
    exog_val = fourier_exog(val_index)

    # No seasonal integration (D=0). Weekly signal is carried by dow
    # dummies inside the exog; yearly signal by Fourier. Drift is avoided
    # by d=1 on the ARIMA side only.
    model = SARIMAX(
        log_y,
        exog=exog_train,
        order=(2, 1, 2),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=300, method="lbfgs")
    fcast = res.forecast(steps=len(val_index), exog=exog_val)
    pred = np.expm1(np.asarray(fcast, dtype=float))
    # Guard against runaway point forecasts in regime-break series.
    cap = 3.0 * float(train_y.max())
    return np.clip(pred, 0.0, cap)


def fit_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    random_state: int = 42,
) -> tuple[np.ndarray, "lightgbm.Booster"]:  # type: ignore[name-defined]
    """LightGBM on log(target). Uses all feature_cols; missing values are
    left to LightGBM's native NaN handling. Returns predictions in original
    scale.
    """
    import lightgbm as lgb

    X_tr = train_df[feature_cols]
    y_tr = np.log1p(train_df[target].values)
    X_va = val_df[feature_cols]

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": random_state,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    model = lgb.train(params, dtrain, num_boost_round=1500)
    preds_log = model.predict(X_va)
    preds = np.expm1(preds_log)
    return np.clip(preds, 0, None), model


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #


FOLDS: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = [
    # (train_start, train_end, val_start, val_end)
    (REGIME_START, pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-30")),
    (REGIME_START, pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-30")),
]


def rolling_cv(
    panel: pd.DataFrame,
    target: str,
    folds: list[tuple] = FOLDS,
    return_fold_preds: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Run all candidate models on each fold; return a long-format dataframe
    of metrics (one row per model × fold) plus a 'Uplift_vs_naive_%' column.

    The ensemble weights are tuned per-target by minimizing the
    concatenated-fold MAE on a 3-model convex grid (LGBM, SARIMAX, naive).
    """
    feats, feature_cols = add_features(panel, target)

    stash: dict = {"folds": {}, "feature_cols": feature_cols}
    rows = []
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(folds, start=1):
        train_mask = (feats.index >= tr_s) & (feats.index <= tr_e)
        val_mask = (feats.index >= va_s) & (feats.index <= va_e)
        train_df = feats.loc[train_mask].dropna(subset=[target])
        val_df = feats.loc[val_mask]
        assert len(val_df) == 548, f"Fold {i} val_len={len(val_df)} != 548"

        y_val = val_df[target].values

        preds_naive = seasonal_naive(feats[target].loc[:tr_e], val_df.index)
        rows.append(dict(fold=i, model="seasonal_naive",
                         **compute_metrics(y_val, preds_naive)))

        try:
            preds_sar = fit_sarimax(train_df[target], val_df.index, feats[target])
            m_sar = compute_metrics(y_val, preds_sar)
        except Exception as e:
            print(f"SARIMAX failed fold {i}: {e}")
            preds_sar = preds_naive.copy()
            m_sar = dict.fromkeys(("MAE", "RMSE", "R2"), float("nan"))
        rows.append(dict(fold=i, model="sarimax", **m_sar))

        preds_lgb, _ = fit_lgbm(train_df, val_df, feature_cols, target)
        rows.append(dict(fold=i, model="lightgbm",
                         **compute_metrics(y_val, preds_lgb)))

        stash["folds"][i] = {
            "y_true": y_val,
            "naive": preds_naive,
            "sarimax": preds_sar,
            "lgbm": preds_lgb,
            "val_index": val_df.index,
        }

    # --- Tune ensemble weights on concatenated fold predictions ---------- #
    T = np.concatenate([f["y_true"] for f in stash["folds"].values()])
    L = np.concatenate([f["lgbm"] for f in stash["folds"].values()])
    S = np.concatenate([f["sarimax"] for f in stash["folds"].values()])
    N = np.concatenate([f["naive"] for f in stash["folds"].values()])
    best_mae, best_w = float("inf"), (1.0, 0.0, 0.0)
    for wl in np.arange(0.4, 1.01, 0.05):
        for ws in np.arange(0.0, 1.0 - wl + 1e-9, 0.05):
            wn = 1.0 - wl - ws
            if wn < -1e-9:
                continue
            mae = float(np.mean(np.abs(T - (wl * L + ws * S + wn * N))))
            if mae < best_mae:
                best_mae, best_w = mae, (float(wl), float(ws), float(wn))
    stash["ensemble_weights"] = {"lgbm": best_w[0], "sarimax": best_w[1], "naive": best_w[2]}

    # Record per-fold ensemble metrics with the tuned weights.
    for i, f in stash["folds"].items():
        preds_ens = (
            best_w[0] * f["lgbm"] + best_w[1] * f["sarimax"] + best_w[2] * f["naive"]
        )
        rows.append(dict(fold=i, model="ensemble_tuned",
                         **compute_metrics(f["y_true"], preds_ens)))

    results = pd.DataFrame(rows)
    naive_mae = results[results["model"] == "seasonal_naive"].set_index("fold")["MAE"]
    results["Uplift_MAE_%"] = results.apply(
        lambda r: 100.0 * (naive_mae[r["fold"]] - r["MAE"]) / naive_mae[r["fold"]],
        axis=1,
    )
    if return_fold_preds:
        return results, stash
    return results


# --------------------------------------------------------------------------- #
# Final training & inference
# --------------------------------------------------------------------------- #


def fit_final_and_predict(
    panel: pd.DataFrame,
    sample_sub: pd.DataFrame,
    target: str,
    ensemble_weights: dict,
) -> tuple[np.ndarray, "lightgbm.Booster", list[str]]:  # type: ignore[name-defined]
    """Train LightGBM + SARIMAX on the full post-regime window
    (2019-01-01 → 2022-12-31), then predict the 548 test days.

    Because the test period extends into 2024, direct application of
    feature `Revenue_lag_364` at test dates would point into either
    the training window (for 2023 dates) or into the test window itself
    (for 2024 dates). We handle this with *recursive* substitution:
    predict 2023 first with lag-364 from 2022, then use those 2023
    predictions to supply lag-364/365 for 2024 dates.
    """
    tr_s, tr_e = REGIME_START, TRAIN_END
    # ---- Build a union panel (train + test) where only the train
    # portion has observed targets. Test portion has NaN targets until
    # we recursively fill them.
    union_idx = pd.date_range(panel.index.min(), sample_sub["Date"].max(), freq="D")
    full_panel = panel.reindex(union_idx).copy()
    full_panel.index.name = "Date"

    feats, feature_cols = add_features(full_panel, target)

    # LightGBM on full post-regime training window
    train_mask = (feats.index >= tr_s) & (feats.index <= tr_e)
    train_df = feats.loc[train_mask].dropna(subset=[target])

    import lightgbm as lgb

    X_tr = train_df[feature_cols]
    y_tr = np.log1p(train_df[target].values)
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=1500)

    # Recursive inference, optimized: only a handful of feature columns
    # depend on the target values of test-window dates. Specifically:
    #   - {target}_lag_364 on dates ≥ TEST_START + 364 days
    #   - {target}_lag_365 on dates ≥ TEST_START + 365 days
    #   - {target}_lag_728 never (728 days before a 2023-01-01 point is
    #       2021-01-03, still in train — safe)
    #   - {target}_roll_mean_W_lag364 windows that slide through the test
    #       window.
    # We pre-compute features for ALL test dates with the observed train
    # target, then on each test date after predicting, we update
    # `y_series` and recompute ONLY the lag/rolling target features for
    # the affected future dates (dates d' where the target at d is used
    # in d's lag feature).
    test_dates = pd.DatetimeIndex(sample_sub["Date"]).sort_values()
    preds_lgb = pd.Series(index=test_dates, dtype=float)

    # y_series: observed train + empty test; we fill test as we go.
    y_series = full_panel[target].copy()

    # Precompute features using current y_series once.
    feats_work = feats.copy()

    # Which lag/rolling columns reference the target? These are the only
    # ones we need to refresh after each prediction.
    lag_cols = {L: f"{target}_lag_{L}" for L in (364, 365, 728) if f"{target}_lag_{L}" in feats_work.columns}
    roll_cols = [c for c in feats_work.columns if c.startswith(f"{target}_roll_")]

    for d in test_dates:
        row = feats_work.loc[[d], feature_cols]
        pred_log = float(model.predict(row)[0])
        pred_val = max(float(np.expm1(pred_log)), 0.0)
        preds_lgb.loc[d] = pred_val
        y_series.loc[d] = pred_val  # feed back

        # Refresh lag columns that now reference d for future rows.
        for L, col in lag_cols.items():
            future_d = d + pd.Timedelta(days=L)
            if future_d in feats_work.index:
                feats_work.at[future_d, col] = pred_val
        # Rolling means/stds over a backward window of y.shift(364). Simplest:
        # recompute the affected columns for all future dates once per step
        # would be slow; instead, we recompute only if d impacts any future
        # rolling window — which it does whenever d - 364 + offset falls in
        # the rolling window. Re-derive the full rolling series cheaply
        # from y_series.shift(364).
        if roll_cols:
            shifted = y_series.shift(364)
            for W in (7, 28, 91, 364):
                m_col = f"{target}_roll_mean_{W}_lag364"
                s_col = f"{target}_roll_std_{W}_lag364"
                if m_col in feats_work.columns:
                    feats_work[m_col] = shifted.rolling(W, min_periods=1).mean()
                if s_col in feats_work.columns:
                    feats_work[s_col] = shifted.rolling(W, min_periods=1).std()

    # SARIMAX final fit on the same window, forecast 548 days ahead
    try:
        preds_sar = fit_sarimax(
            train_df[target], test_dates, full_panel[target]
        )
    except Exception as e:  # pragma: no cover
        print(f"Final SARIMAX failed: {e}")
        preds_sar = preds_lgb.values.copy()

    # Seasonal-naive final: look back 364 days from test dates into the
    # observed training series (y_series already contains LGBM preds, but
    # for naive we want purely observed data, so use original panel).
    preds_naive = seasonal_naive(full_panel[target].loc[:tr_e], test_dates)

    # Combine with CV-tuned weights
    w = ensemble_weights
    preds_final = (
        w["lgbm"] * preds_lgb.values
        + w["sarimax"] * np.asarray(preds_sar, dtype=float)
        + w["naive"] * np.asarray(preds_naive, dtype=float)
    )
    preds_final = np.clip(preds_final, 0.0, None)
    return preds_final, model, feature_cols


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """LightGBM gain-based importance table, sorted descending."""
    gains = model.feature_importance(importance_type="gain")
    splits = model.feature_importance(importance_type="split")
    df = pd.DataFrame(
        {"feature": feature_cols, "gain": gains, "split": splits}
    ).sort_values("gain", ascending=False).reset_index(drop=True)
    return df
