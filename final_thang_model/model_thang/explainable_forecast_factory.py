"""Explainable forecast factory for the Datathon Revenue/COGS task.

This pipeline is designed for the technical-report part of the score as well as
leaderboard performance:

- direct multi-step horizon model: train rows mimic "forecast the next 548 days
  from a historical cutoff"
- LightGBM component for nonlinear calendar/lag effects
- Ridge component for smooth trend/seasonality
- deterministic holiday/calendar features derived from Gregorian dates
- train-only regime level calibration
- feature importance, SHAP summary, feature-group audit, and a markdown report

It never reads `Revenue` or `COGS` from `sample_submission.csv`; the final
horizon dates are constructed directly from the contest specification.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_thang.forecast_pipeline import (  # noqa: E402
    FORECAST_END,
    FORECAST_START,
    POST_REGIME_START,
    TARGETS,
    TRAIN_END,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)


DATA = ROOT.parent / "data"
DOCS = ROOT / "docs"
OUT = ROOT / "model_thang" / "artifacts"
MODEL_OUT = OUT / "saved_models" / "direct_factory"
HOLIDAY_CSV = DOCS / "vietnam_calendar_events_deterministic_2012_2024.csv"

SEED = 20260429
DIRECT_CUTOFFS = pd.to_datetime(
    [
        "2014-12-31",
        "2015-06-30",
        "2015-12-31",
        "2016-06-30",
        "2016-12-31",
        "2017-06-30",
        "2017-12-31",
        "2018-06-30",
        "2018-12-31",
        "2019-06-30",
        "2019-12-31",
        "2020-06-30",
        "2020-12-31",
        "2021-06-30",
        "2021-07-01",
    ]
)
CV_CUTOFFS = pd.to_datetime(["2020-06-30", "2020-12-31", "2021-07-01"])
LAGS = (7, 14, 28, 56, 91, 182, 364, 365, 371, 548, 728, 730)
ROLLS = (7, 28, 56, 91, 182, 365)


@dataclass
class FittedDirect:
    target: str
    feature_cols: list[str]
    lgb_model: object
    ridge_model: Pipeline
    weights: dict[str, float]
    cv_metrics: pd.DataFrame
    importance: pd.DataFrame
    shap_importance: pd.DataFrame


def save_fitted_direct(fitted: FittedDirect, model_dir: Path) -> dict[str, str]:
    """Persist trained direct models and feature metadata for inference."""
    target_dir = model_dir / fitted.target
    target_dir.mkdir(parents=True, exist_ok=True)

    lgb_path = target_dir / "lightgbm.txt"
    ridge_path = target_dir / "ridge_pipeline.pkl"
    metadata_path = target_dir / "metadata.json"

    fitted.lgb_model.save_model(str(lgb_path))
    with ridge_path.open("wb") as f:
        pickle.dump(fitted.ridge_model, f)
    metadata = {
        "target": fitted.target,
        "feature_cols": fitted.feature_cols,
        "weights": fitted.weights,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        f"{fitted.target}_lgb_model": str(lgb_path),
        f"{fitted.target}_ridge_model": str(ridge_path),
        f"{fitted.target}_metadata": str(metadata_path),
    }


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _nearest_event_distance(dates: pd.DatetimeIndex, events: list[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
    event_ts = np.asarray([e.value for e in sorted(events)], dtype=np.int64)
    date_ts = dates.values.astype("datetime64[ns]").astype(np.int64)
    days_to = np.full(len(dates), 180.0)
    days_since = np.full(len(dates), 180.0)
    for i, d in enumerate(date_ts):
        diffs = (event_ts - d) / 86_400_000_000_000
        future = diffs[diffs >= 0]
        past = diffs[diffs <= 0]
        if len(future):
            days_to[i] = min(float(future.min()), 180.0)
        if len(past):
            days_since[i] = min(float(-past.max()), 180.0)
    return days_to, days_since


def load_holiday_events() -> dict[str, list[pd.Timestamp]]:
    table = pd.read_csv(HOLIDAY_CSV)
    events: dict[str, list[pd.Timestamp]] = {}
    for col in table.columns:
        if col == "year":
            continue
        events[_safe_name(col)] = pd.to_datetime(table[col]).dropna().tolist()
    return events


def deterministic_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    feats = pd.DataFrame(index=index)
    feats.index.name = "Date"
    feats["forecast_year"] = index.year
    feats["month"] = index.month
    feats["day"] = index.day
    feats["dow"] = index.dayofweek
    feats["doy"] = index.dayofyear
    feats["week"] = index.isocalendar().week.astype(int).to_numpy()
    feats["quarter"] = index.quarter
    feats["is_weekend"] = (index.dayofweek >= 5).astype(int)
    feats["is_month_start"] = index.is_month_start.astype(int)
    feats["is_month_end"] = index.is_month_end.astype(int)
    feats["is_payday_window"] = ((index.day <= 5) | (index.day >= 25)).astype(int)
    feats["is_midmonth_window"] = ((index.day >= 13) & (index.day <= 17)).astype(int)

    t = np.arange(len(index), dtype=float)
    for period, prefix, harmonics in ((7.0, "week", 3), (365.25, "year", 8)):
        for k in range(1, harmonics + 1):
            feats[f"sin_{prefix}_{k}"] = np.sin(2 * np.pi * k * t / period)
            feats[f"cos_{prefix}_{k}"] = np.cos(2 * np.pi * k * t / period)

    try:
        from src.calendar_vn import add_vn_calendar

        vn = add_vn_calendar(pd.DataFrame({"Date": index})).set_index("Date")
        for col in vn.columns:
            feats[f"vn_{col}"] = vn[col].to_numpy()
    except Exception as exc:  # pragma: no cover - only if local calendar breaks
        feats["vn_calendar_error"] = 1
        feats["vn_calendar_error_code"] = float(abs(hash(str(exc))) % 1000)

    # Explicit features from a generated deterministic calendar table. The table
    # is reproducible from Gregorian Date rules and does not encode external
    # year-specific holiday lookup data.
    for name, dates in load_holiday_events().items():
        days_to, days_since = _nearest_event_distance(index, dates)
        feats[f"hol_days_to_{name}"] = days_to
        feats[f"hol_days_since_{name}"] = days_since
        feats[f"hol_is_{name}"] = (days_since == 0).astype(int)
        feats[f"hol_pre7_{name}"] = ((days_to >= 0) & (days_to <= 7)).astype(int)
        feats[f"hol_post3_{name}"] = ((days_since >= 0) & (days_since <= 3)).astype(int)

    # Coarser business-season flags are easier to explain than dozens of exact
    # dates and often more stable in validation.
    feats["season_q1_tet_low"] = feats["quarter"].eq(1).astype(int)
    feats["season_q2_peak"] = feats["month"].isin([4, 5, 6]).astype(int)
    feats["season_q4_promo"] = feats["month"].isin([11, 12]).astype(int)
    feats["season_summer_transition"] = feats["month"].isin([7, 8, 9]).astype(int)
    return feats.replace([np.inf, -np.inf], np.nan)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
    }


def cutoff_priors(y: pd.Series, cutoff: pd.Timestamp) -> dict[str, object]:
    hist = y.loc[:cutoff].dropna()
    recent_start = max(hist.index.min(), cutoff - pd.Timedelta(days=365 * 4))
    recent = hist.loc[recent_start:cutoff]
    if len(recent) < 365:
        recent = hist

    by_doy = recent.groupby(recent.index.dayofyear)
    by_month = recent.groupby(recent.index.month)
    by_dow = recent.groupby(recent.index.dayofweek)
    by_md = recent.groupby([recent.index.month, recent.index.dayofweek])

    full_years = hist.groupby(hist.index.year).filter(lambda s: len(s) >= 360)
    annual = full_years.groupby(full_years.index.year).mean() if len(full_years) else hist.groupby(hist.index.year).mean()
    pre = annual.loc[annual.index <= min(2018, cutoff.year)].mean() if len(annual) else hist.mean()
    post = annual.loc[annual.index >= 2019].mean() if (len(annual) and (annual.index >= 2019).any()) else hist.mean()

    return {
        "hist_mean": float(hist.mean()),
        "recent_mean": float(recent.mean()),
        "recent_median": float(recent.median()),
        "doy_mean": by_doy.mean(),
        "doy_median": by_doy.median(),
        "month_mean": by_month.mean(),
        "dow_mean": by_dow.mean(),
        "month_dow_mean": by_md.mean(),
        "annual_mean": float(annual.iloc[-1]) if len(annual) else float(hist.mean()),
        "pre_break_mean": float(pre),
        "post_break_mean": float(post),
    }


def _series_get(y: pd.Series, date: pd.Timestamp, cutoff: pd.Timestamp | None = None) -> float:
    if cutoff is not None and date > cutoff:
        return np.nan
    val = y.get(date, np.nan)
    return float(val) if pd.notna(val) else np.nan


def build_feature_rows(
    sales: pd.DataFrame,
    target: str,
    cutoffs: pd.DatetimeIndex,
    include_target: bool,
    final_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    y = sales.set_index("Date")[target].sort_index()
    min_date = y.index.min()
    max_feature_date = FORECAST_END if final_dates is not None else TRAIN_END
    calendar = deterministic_calendar(pd.date_range(min_date, max_feature_date, freq="D"))

    rows = []
    prior_cache = {cutoff: cutoff_priors(y, cutoff) for cutoff in cutoffs}
    for cutoff in cutoffs:
        if final_dates is None:
            dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=548, freq="D")
            dates = dates[dates <= TRAIN_END]
        else:
            dates = final_dates
        priors = prior_cache[cutoff]
        hist = y.loc[:cutoff].dropna()
        for forecast_date in dates:
            if include_target and forecast_date > TRAIN_END:
                continue
            row: dict[str, float | str | pd.Timestamp] = {
                "cutoff": cutoff,
                "Date": forecast_date,
                "horizon": int((forecast_date - cutoff).days),
                "cutoff_year": int(cutoff.year),
                "cutoff_month": int(cutoff.month),
                "cutoff_dow": int(cutoff.dayofweek),
                "cutoff_after_2019": int(cutoff >= POST_REGIME_START),
            }
            cal = calendar.loc[forecast_date].to_dict()
            row.update(cal)

            horizon = row["horizon"]
            row["horizon_le_90"] = int(horizon <= 90)
            row["horizon_91_182"] = int(91 <= horizon <= 182)
            row["horizon_183_365"] = int(183 <= horizon <= 365)
            row["horizon_gt_365"] = int(horizon > 365)

            for lag in LAGS:
                row[f"{target}_forecast_lag{lag}_known"] = _series_get(
                    y, forecast_date - pd.Timedelta(days=lag), cutoff
                )
                row[f"{target}_anchor_lag{lag}"] = _series_get(
                    y, cutoff - pd.Timedelta(days=lag)
                )

            for window in ROLLS:
                tail = hist.tail(window)
                row[f"{target}_anchor_roll_mean{window}"] = float(tail.mean()) if len(tail) else np.nan
                row[f"{target}_anchor_roll_std{window}"] = float(tail.std()) if len(tail) > 2 else np.nan
                row[f"{target}_anchor_roll_median{window}"] = float(tail.median()) if len(tail) else np.nan

            latest = _series_get(y, cutoff)
            lag365 = _series_get(y, cutoff - pd.Timedelta(days=365))
            lag730 = _series_get(y, cutoff - pd.Timedelta(days=730))
            row[f"{target}_anchor_latest"] = latest
            row[f"{target}_anchor_yoy_ratio365"] = latest / lag365 if lag365 and not np.isnan(lag365) else np.nan
            row[f"{target}_anchor_2y_ratio730"] = latest / lag730 if lag730 and not np.isnan(lag730) else np.nan
            row[f"{target}_recent_vs_hist"] = priors["recent_mean"] / priors["hist_mean"] if priors["hist_mean"] else np.nan
            row[f"{target}_annual_mean_cutoff"] = priors["annual_mean"]
            row[f"{target}_pre_break_mean_cutoff"] = priors["pre_break_mean"]
            row[f"{target}_post_break_mean_cutoff"] = priors["post_break_mean"]
            row[f"{target}_post_to_pre_ratio"] = (
                priors["post_break_mean"] / priors["pre_break_mean"] if priors["pre_break_mean"] else np.nan
            )

            doy = int(forecast_date.dayofyear)
            month = int(forecast_date.month)
            dow = int(forecast_date.dayofweek)
            md_key = (month, dow)
            row[f"{target}_doy_mean_cutoff"] = float(priors["doy_mean"].get(doy, priors["recent_mean"]))
            row[f"{target}_doy_median_cutoff"] = float(priors["doy_median"].get(doy, priors["recent_median"]))
            row[f"{target}_month_mean_cutoff"] = float(priors["month_mean"].get(month, priors["recent_mean"]))
            row[f"{target}_dow_mean_cutoff"] = float(priors["dow_mean"].get(dow, priors["recent_mean"]))
            row[f"{target}_month_dow_mean_cutoff"] = float(
                priors["month_dow_mean"].get(md_key, priors["recent_mean"])
            )
            row[f"{target}_doy_to_recent"] = row[f"{target}_doy_mean_cutoff"] / priors["recent_mean"]
            row[f"{target}_month_to_recent"] = row[f"{target}_month_mean_cutoff"] / priors["recent_mean"]

            if include_target:
                row[target] = _series_get(y, forecast_date)
            rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    exclude = {"Date", "cutoff", target}
    return [
        c
        for c in frame.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]


def train_lgb(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], target: str, rounds: int | None = None):
    import lightgbm as lgb

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 3,
        "lambda_l1": 0.2,
        "lambda_l2": 1.0,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(train[features], label=np.log1p(train[target].clip(lower=0).to_numpy()))
    if rounds is not None:
        return lgb.train(params, dtrain, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])

    dvalid = lgb.Dataset(valid[features], label=np.log1p(valid[target].clip(lower=0).to_numpy()), reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )


def train_ridge(train: pd.DataFrame, features: list[str], target: str) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 13))),
        ]
    )
    model.fit(train[features], np.log1p(train[target].clip(lower=0).to_numpy()))
    return model


def predict_lgb(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.expm1(np.asarray(model.predict(frame[features]), dtype=float)).clip(min=1.0)


def predict_ridge(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.expm1(model.predict(frame[features])).clip(min=1.0)


def tune_component_weights(actual: np.ndarray, preds: dict[str, np.ndarray], step: float = 0.05) -> dict[str, float]:
    names = list(preds)
    best = (float("inf"), None)
    grid = np.arange(0, 1 + 1e-9, step)
    if len(names) != 3:
        raise ValueError("This helper expects exactly three components")
    for w0 in grid:
        for w1 in grid:
            if w0 + w1 > 1:
                continue
            weights = [w0, w1, 1 - w0 - w1]
            pred = sum(weights[i] * preds[names[i]] for i in range(3))
            mae = mean_absolute_error(actual, pred)
            if mae < best[0]:
                best = (float(mae), weights)
    assert best[1] is not None
    return {names[i]: float(best[1][i]) for i in range(3)}


def feature_group(feature: str) -> str:
    if feature == "horizon" or feature.startswith("horizon_"):
        return "horizon"
    if feature in {"forecast_year", "month", "day", "dow", "doy", "week", "quarter"}:
        return "calendar_seasonality"
    if feature.startswith("cutoff_") or "break" in feature or "recent_vs_hist" in feature:
        return "regime_level"
    if feature.startswith("hol_") or feature.startswith("vn_") or feature.startswith("season_"):
        return "holiday_event"
    if any(k in feature for k in ["month", "dow", "doy", "week", "quarter", "sin_", "cos_", "day"]):
        return "calendar_seasonality"
    if "forecast_lag" in feature or "anchor_lag" in feature:
        return "target_lag"
    if "ratio" in feature and "anchor" in feature:
        return "target_lag"
    if "roll_" in feature or "anchor_latest" in feature or "annual_mean" in feature:
        return "anchor_level"
    if "cutoff" in feature:
        return "regime_level"
    if "mean_cutoff" in feature or "median_cutoff" in feature or "_to_recent" in feature:
        return "seasonal_prior"
    return "other"


def cv_and_fit(sales: pd.DataFrame, target: str) -> FittedDirect:
    train_frame = build_feature_rows(sales, target, DIRECT_CUTOFFS, include_target=True)
    features = feature_columns(train_frame, target)
    rows = []
    oof_actual = []
    oof_preds = {"lgb": [], "ridge": [], "doy_prior": []}
    best_iters = []

    for val_cutoff in CV_CUTOFFS:
        fit = train_frame[train_frame["cutoff"] < val_cutoff].copy()
        val = train_frame[train_frame["cutoff"] == val_cutoff].copy()
        if fit.empty or val.empty:
            continue
        # Use the previous cutoff as early-stopping validation if available.
        inner_valid_cutoff = fit["cutoff"].max()
        inner_train = fit[fit["cutoff"] < inner_valid_cutoff]
        inner_valid = fit[fit["cutoff"] == inner_valid_cutoff]
        if inner_train.empty or inner_valid.empty:
            inner_train, inner_valid = fit.iloc[:-min(548, len(fit) // 4)], fit.iloc[-min(548, len(fit) // 4):]

        lgb_model = train_lgb(inner_train, inner_valid, features, target)
        best_iters.append(int(lgb_model.best_iteration or 900))
        ridge_model = train_ridge(fit, features, target)

        pred_lgb = predict_lgb(lgb_model, val, features)
        pred_ridge = predict_ridge(ridge_model, val, features)
        pred_doy = val[f"{target}_doy_mean_cutoff"].to_numpy(dtype=float)
        actual = val[target].to_numpy(dtype=float)

        for name, pred in (("lgb", pred_lgb), ("ridge", pred_ridge), ("doy_prior", pred_doy)):
            rows.append(
                {
                    "target": target,
                    "fold_cutoff": str(val_cutoff.date()),
                    "model": name,
                    **metrics(actual, pred),
                }
            )
        oof_actual.append(actual)
        oof_preds["lgb"].append(pred_lgb)
        oof_preds["ridge"].append(pred_ridge)
        oof_preds["doy_prior"].append(pred_doy)

    actual_all = np.concatenate(oof_actual)
    pred_all = {name: np.concatenate(values) for name, values in oof_preds.items()}
    weights = tune_component_weights(actual_all, pred_all)
    for val_cutoff in CV_CUTOFFS:
        val_rows = [r for r in rows if r["fold_cutoff"] == str(val_cutoff.date())]
        # Reconstruct from stored arrays by recomputing once for clarity would be
        # wasteful; final CV ensemble metrics are added from concatenated OOF.
        if not val_rows:
            continue
    blended_all = sum(weights[name] * pred_all[name] for name in weights)
    rows.append({"target": target, "fold_cutoff": "all_oof", "model": "weighted_direct", **metrics(actual_all, blended_all)})
    cv_metrics = pd.DataFrame(rows)

    full_rounds = max(300, int(np.median(best_iters) * 1.08)) if best_iters else 900
    full_model = train_lgb(train_frame, train_frame.tail(548), features, target, rounds=full_rounds)
    full_ridge = train_ridge(train_frame, features, target)

    gains = full_model.feature_importance(importance_type="gain")
    importance = (
        pd.DataFrame({"target": target, "feature": features, "gain": gains})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    importance["group"] = importance["feature"].map(feature_group)

    shap_imp = compute_shap_importance(full_model, train_frame, features, target)
    return FittedDirect(target, features, full_model, full_ridge, weights, cv_metrics, importance, shap_imp)


def compute_shap_importance(model, train_frame: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    try:
        import shap

        sample = train_frame[features].sample(min(2000, len(train_frame)), random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(sample, check_additivity=False)
        mean_abs = np.abs(np.asarray(values)).mean(axis=0)
        out = pd.DataFrame({"target": target, "feature": features, "mean_abs_shap": mean_abs})
        out["group"] = out["feature"].map(feature_group)
        return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception as exc:
        return pd.DataFrame(
            [{"target": target, "feature": "__shap_failed__", "mean_abs_shap": 0.0, "group": str(exc)[:120]}]
        )


def predict_final_component(sales: pd.DataFrame, fitted: FittedDirect) -> pd.DataFrame:
    dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    frame = build_feature_rows(
        sales,
        fitted.target,
        pd.DatetimeIndex([TRAIN_END]),
        include_target=False,
        final_dates=dates,
    )
    pred_lgb = predict_lgb(fitted.lgb_model, frame, fitted.feature_cols)
    pred_ridge = predict_ridge(fitted.ridge_model, frame, fitted.feature_cols)
    pred_doy = frame[f"{fitted.target}_doy_mean_cutoff"].to_numpy(dtype=float)
    pred = (
        fitted.weights.get("lgb", 0) * pred_lgb
        + fitted.weights.get("ridge", 0) * pred_ridge
        + fitted.weights.get("doy_prior", 0) * pred_doy
    )
    return pd.DataFrame(
        {
            "Date": dates,
            fitted.target: pred.clip(min=1.0),
            f"{fitted.target}_lgb": pred_lgb,
            f"{fitted.target}_ridge": pred_ridge,
            f"{fitted.target}_doy_prior": pred_doy,
        }
    )


def blend_submissions(base: pd.DataFrame, direct: pd.DataFrame, direct_weight: float) -> pd.DataFrame:
    out = base[["Date"]].copy()
    for target in TARGETS:
        out[target] = (1.0 - direct_weight) * base[target].to_numpy() + direct_weight * direct[target].to_numpy()
    return out


def write_report(
    cv: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    group_importance: pd.DataFrame,
    shap_group: pd.DataFrame,
    files: dict[str, str],
) -> None:
    lines = [
        "# Explainable Forecast Factory Report",
        "",
        "## Objective",
        "",
        "Forecast daily `Revenue` and `COGS` for 2023-01-01 to 2024-07-01 with a reproducible, leakage-safe pipeline.",
        "",
        "## Scoring Alignment",
        "",
        "- Leaderboard: produces Kaggle-ready submissions and blends against the current best no-sample candidate.",
        "- Technical report: uses time-aware CV, explicit holiday audit features, LightGBM gain importance, SHAP mean absolute importance, and feature-group business explanations.",
        "",
        "## Leakage Policy",
        "",
        "- Does not read target values from `sample_submission.csv`.",
        "- Future dates use public calendar features, historical target lags known at each cutoff, and train-cutoff seasonal priors only.",
        "- Pseudo-horizon rows mimic historical forecast cutoffs rather than using same-day future operational aggregates.",
        "- Yearly calibration uses only `sales.csv` historical regime assumptions.",
        "",
        "## Time-Aware CV",
        "",
        cv.round(4).to_markdown(index=False),
        "",
        "## Direct Component Weights",
        "",
        pd.DataFrame(
            [{"target": t, **w} for t, w in weights.items()]
        ).round(4).to_markdown(index=False),
        "",
        "## Top Feature Groups By LightGBM Gain",
        "",
        group_importance.round(4).to_markdown(index=False),
        "",
        "## Top Feature Groups By SHAP",
        "",
        shap_group.round(6).to_markdown(index=False),
        "",
        "## Business Explanation",
        "",
        "- `seasonal_prior`: the model relies on day-of-year/month-weekday priors because fashion demand has stable annual shape and the data generator preserves strong yearly recurrence.",
        "- `target_lag` and `anchor_level`: recent known demand level and one-/two-year memory anchor the forecast to the latest business regime.",
        "- `holiday_event`: Tet, Hung Kings, Apr 30-May 1, 11/11, Black Friday, 12/12 and gifting days identify demand or logistics disruption windows.",
        "- `regime_level`: post-2019 structural break and 2022 recovery are modeled explicitly before calibration.",
        "- `horizon`: the direct model learns different behavior for short, medium, and >365-day forecasts, reducing recursive drift.",
        "",
        "## Generated Files",
        "",
    ]
    for name, path in files.items():
        lines.append(f"- `{name}`: `{path}`")
    (OUT / "explainable_forecast_factory_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    sales = load_sales()

    fitted: dict[str, FittedDirect] = {}
    direct_parts = []
    all_cv = []
    all_imp = []
    all_shap = []
    weights = {}
    model_files: dict[str, str] = {}

    for target in TARGETS:
        fit = cv_and_fit(sales, target)
        fitted[target] = fit
        model_files.update(save_fitted_direct(fit, MODEL_OUT))
        weights[target] = fit.weights
        all_cv.append(fit.cv_metrics)
        all_imp.append(fit.importance)
        all_shap.append(fit.shap_importance)
        direct_parts.append(predict_final_component(sales, fit))

    dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    direct = pd.DataFrame({"Date": dates})
    component_debug = pd.DataFrame({"Date": dates})
    for part in direct_parts:
        target = next(t for t in TARGETS if t in part.columns)
        direct[target] = part[target].to_numpy()
        for col in part.columns:
            if col != "Date" and col != target:
                component_debug[col] = part[col].to_numpy()

    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}
    direct_regime = normalise_yearly(direct, levels)

    files: dict[str, str] = {}
    direct_raw_path = OUT / "submission_direct_factory_raw.csv"
    direct_regime_path = OUT / "submission_direct_factory_regime_recovery.csv"
    export_submission(direct, direct_raw_path)
    export_submission(direct_regime, direct_regime_path)
    files["direct_raw"] = str(direct_raw_path)
    files["direct_regime"] = str(direct_regime_path)
    component_debug.to_csv(OUT / "direct_factory_component_debug.csv", index=False)
    files["component_debug"] = str(OUT / "direct_factory_component_debug.csv")

    base_path = OUT / "m5b50300515.csv"
    if base_path.exists():
        base = pd.read_csv(base_path, parse_dates=["Date"])
        for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            blended = blend_submissions(base, direct_regime, w)
            path = OUT / f"submission_m5_direct_blend_{int((1-w)*100):02d}_{int(w*100):02d}.csv"
            export_submission(blended, path)
            files[f"m5_direct_{int((1-w)*100):02d}_{int(w*100):02d}"] = str(path)

    cv = pd.concat(all_cv, ignore_index=True)
    imp = pd.concat(all_imp, ignore_index=True)
    shap_imp = pd.concat(all_shap, ignore_index=True)
    group_importance = (
        imp.groupby(["target", "group"], as_index=False)["gain"]
        .sum()
        .sort_values(["target", "gain"], ascending=[True, False])
    )
    shap_group = (
        shap_imp.groupby(["target", "group"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values(["target", "mean_abs_shap"], ascending=[True, False])
    )

    cv.to_csv(OUT / "direct_factory_cv_metrics.csv", index=False)
    imp.to_csv(OUT / "direct_factory_feature_importance.csv", index=False)
    shap_imp.to_csv(OUT / "direct_factory_shap_importance.csv", index=False)
    group_importance.to_csv(OUT / "direct_factory_feature_group_importance.csv", index=False)
    shap_group.to_csv(OUT / "direct_factory_shap_group_importance.csv", index=False)
    files["cv_metrics"] = str(OUT / "direct_factory_cv_metrics.csv")
    files["feature_importance"] = str(OUT / "direct_factory_feature_importance.csv")
    files["shap_importance"] = str(OUT / "direct_factory_shap_importance.csv")
    files["feature_group_importance"] = str(OUT / "direct_factory_feature_group_importance.csv")

    audit = {
        "cv_summary": cv.to_dict(orient="records"),
        "weights": weights,
        "levels": levels,
        "files": files,
        "model_files": model_files,
        "holiday_source": str(HOLIDAY_CSV),
        "recommended_low_risk_submit": files.get("m5_direct_85_15", files["direct_regime"]),
        "recommended_next_probe": files.get("m5_direct_80_20", files["direct_regime"]),
        "recommended_direct_submit": files["direct_regime"],
        "leakage_policy": [
            "no sample_submission target values read",
            "future operational aggregates are not used",
            "holiday features are deterministic transforms of Gregorian Date, including solar-to-lunar conversion",
            "direct rows use only information known at each historical cutoff",
        ],
    }
    (OUT / "direct_factory_audit.json").write_text(json.dumps(audit, indent=2, default=float))
    files["audit"] = str(OUT / "direct_factory_audit.json")
    write_report(cv, weights, group_importance, shap_group, files)
    files["report"] = str(OUT / "explainable_forecast_factory_report.md")

    print(json.dumps(audit, indent=2, default=float))


if __name__ == "__main__":
    main()
