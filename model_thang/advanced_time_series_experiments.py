from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_thang.explainable_forecast_factory import (  # noqa: E402
    DIRECT_CUTOFFS,
    TARGETS,
    build_feature_rows,
    feature_columns,
    metrics,
    predict_lgb,
    predict_ridge,
    train_ridge,
)
from model_thang.forecast_pipeline import (  # noqa: E402
    FORECAST_END,
    FORECAST_START,
    TRAIN_END,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)


warnings.filterwarnings("ignore")

OUT = ROOT / "model_thang" / "artifacts" / "advanced_experiments"
BASE_ARTIFACTS = ROOT / "model_thang" / "artifacts"

VAL_CUTOFF = pd.Timestamp("2020-12-31")
VAL_START = pd.Timestamp("2021-01-01")
VAL_END_USER = pd.Timestamp("2022-07-01")
VAL_END_MATCHED = pd.Timestamp("2022-07-02")
SEED = 20260430

COMPONENTS = ("lgb", "ridge", "doy", "lag730")


@dataclass
class TargetExperiment:
    target: str
    val_frame: pd.DataFrame
    oof_frame: pd.DataFrame
    component_weights: dict[str, float]
    residual_scale: float
    val_predictions: dict[str, np.ndarray]
    oof_predictions: dict[str, np.ndarray]
    features: list[str]


def simplex_weights(names: tuple[str, ...], step: float = 0.05) -> list[dict[str, float]]:
    if len(names) == 1:
        return [{names[0]: 1.0}]
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    out: list[dict[str, float]] = []

    def rec(prefix: list[float], remaining: int, budget: float) -> None:
        if remaining == 1:
            out.append({name: float(w) for name, w in zip(names, [*prefix, budget])})
            return
        for w in grid:
            if w <= budget + 1e-9:
                rec([*prefix, float(w)], remaining - 1, float(round(budget - w, 10)))

    rec([], len(names), 1.0)
    return out


def tune_weights_mae(actual: np.ndarray, preds: dict[str, np.ndarray], step: float = 0.05) -> dict[str, float]:
    names = tuple(preds)
    best_mae = float("inf")
    best_weights: dict[str, float] | None = None
    for weights in simplex_weights(names, step=step):
        pred = sum(weights[name] * preds[name] for name in names)
        mae = float(mean_absolute_error(actual, pred))
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
    assert best_weights is not None
    return best_weights


def weighted_prediction(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights.get(name, 0.0) * pred for name, pred in preds.items())


def lag730_component(frame: pd.DataFrame, target: str) -> np.ndarray:
    lag = frame[f"{target}_forecast_lag730_known"].to_numpy(dtype=float)
    fallback = frame[f"{target}_doy_mean_cutoff"].to_numpy(dtype=float)
    return np.where(np.isfinite(lag), lag, fallback).clip(min=1.0)


def fit_component_models(train_frame: pd.DataFrame, target: str, features: list[str]):
    inner_valid_cutoff = train_frame["cutoff"].max()
    inner_train = train_frame[train_frame["cutoff"] < inner_valid_cutoff].copy()
    inner_valid = train_frame[train_frame["cutoff"] == inner_valid_cutoff].copy()
    if inner_train.empty or inner_valid.empty:
        split = max(1, len(train_frame) - min(548, max(1, len(train_frame) // 4)))
        inner_train = train_frame.iloc[:split].copy()
        inner_valid = train_frame.iloc[split:].copy()
    lgb_model = train_fast_lgb(inner_train, inner_valid, features, target)
    ridge_model = train_ridge(train_frame, features, target)
    return lgb_model, ridge_model


def train_fast_lgb(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], target: str):
    import lightgbm as lgb

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 25,
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
        "num_threads": 4,
    }
    dtrain = lgb.Dataset(train[features], label=np.log1p(train[target].clip(lower=0).to_numpy()))
    dvalid = lgb.Dataset(valid[features], label=np.log1p(valid[target].clip(lower=0).to_numpy()), reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=900,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
    )


def predict_components(lgb_model, ridge_model, frame: pd.DataFrame, features: list[str], target: str) -> dict[str, np.ndarray]:
    return {
        "lgb": predict_lgb(lgb_model, frame, features),
        "ridge": predict_ridge(ridge_model, frame, features),
        "doy": frame[f"{target}_doy_mean_cutoff"].to_numpy(dtype=float).clip(min=1.0),
        "lag730": lag730_component(frame, target),
    }


def make_oof_predictions(train_frame: pd.DataFrame, target: str, features: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cutoffs = [c for c in sorted(train_frame["cutoff"].unique()) if c < np.datetime64(VAL_CUTOFF)]
    # Last folds are closest to the validation/test regime and keep the
    # experiment light enough to iterate inside the contest workspace.
    cutoffs = cutoffs[-5:]
    rows = []
    pred_parts = {name: [] for name in COMPONENTS}
    actual_parts = []

    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        fit = train_frame[train_frame["cutoff"] < cutoff].copy()
        val = train_frame[train_frame["cutoff"] == cutoff].copy()
        if fit.empty or val.empty:
            continue
        lgb_model, ridge_model = fit_component_models(fit, target, features)
        preds = predict_components(lgb_model, ridge_model, val, features, target)
        for name in COMPONENTS:
            pred_parts[name].append(preds[name])
        actual_parts.append(val[target].to_numpy(dtype=float))
        rows.append(val.copy())

    if not rows:
        raise RuntimeError(f"No OOF rows produced for {target}")
    oof_frame = pd.concat(rows, ignore_index=True)
    pred_arrays = {name: np.concatenate(parts) for name, parts in pred_parts.items()}
    pred_arrays["actual"] = np.concatenate(actual_parts)
    return oof_frame, pred_arrays


def residual_features(frame: pd.DataFrame) -> list[str]:
    prefixes = ("hol_", "vn_", "season_")
    explicit = {
        "forecast_year",
        "month",
        "day",
        "dow",
        "doy",
        "week",
        "quarter",
        "is_weekend",
        "is_payday_window",
        "horizon",
        "horizon_le_90",
        "horizon_91_182",
        "horizon_183_365",
        "horizon_gt_365",
    }
    return [
        c
        for c in frame.columns
        if c in explicit or c.startswith(prefixes)
    ]


def fit_residual_model(
    oof_frame: pd.DataFrame,
    residual: np.ndarray,
    val_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    feats = residual_features(oof_frame)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(1, 6, 12))),
        ]
    )
    model.fit(oof_frame[feats], residual)
    oof_corr = model.predict(oof_frame[feats])
    val_corr = model.predict(val_frame[feats])
    return oof_corr, val_corr, float(model.named_steps["ridge"].alpha_), feats


def tune_residual_scale(actual: np.ndarray, base: np.ndarray, corr: np.ndarray) -> float:
    best = (float("inf"), 0.0)
    for scale in np.arange(-0.5, 1.05, 0.05):
        pred = np.clip(base + scale * corr, 1.0, None)
        mae = float(mean_absolute_error(actual, pred))
        if mae < best[0]:
            best = (mae, float(scale))
    return best[1]


def ratio_prior_for_rows(sales: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    ratio = sales.set_index("Date")["COGS"] / sales.set_index("Date")["Revenue"].replace(0, np.nan)
    cache: dict[pd.Timestamp, dict[str, object]] = {}
    out = []
    for cutoff, date in zip(pd.to_datetime(rows["cutoff"]), pd.to_datetime(rows["Date"])):
        cutoff = pd.Timestamp(cutoff)
        if cutoff not in cache:
            hist = ratio.loc[:cutoff].dropna()
            recent = hist.loc[max(hist.index.min(), cutoff - pd.Timedelta(days=365 * 4)): cutoff]
            if len(recent) < 365:
                recent = hist
            cache[cutoff] = {
                "doy": recent.groupby(recent.index.dayofyear).median(),
                "month": recent.groupby(recent.index.month).median(),
                "fallback": float(recent.median()),
            }
        priors = cache[cutoff]
        doy = int(date.dayofyear)
        month = int(date.month)
        value = priors["doy"].get(doy, priors["month"].get(month, priors["fallback"]))
        out.append(float(value))
    return np.asarray(out, dtype=float)


def evaluate_predictions(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
    }


def run_target(sales: pd.DataFrame, target: str) -> TargetExperiment:
    train_cutoffs = pd.DatetimeIndex([c for c in DIRECT_CUTOFFS if c < VAL_CUTOFF])
    train_frame = build_feature_rows(sales, target, train_cutoffs, include_target=True)
    features = feature_columns(train_frame, target)

    oof_frame, oof_preds = make_oof_predictions(train_frame, target, features)
    component_weights = tune_weights_mae(oof_preds["actual"], {k: oof_preds[k] for k in COMPONENTS}, step=0.05)
    oof_blend = weighted_prediction({k: oof_preds[k] for k in COMPONENTS}, component_weights)

    lgb_model, ridge_model = fit_component_models(train_frame, target, features)
    val_dates = pd.date_range(VAL_START, VAL_END_MATCHED, freq="D")
    val_frame = build_feature_rows(
        sales,
        target,
        pd.DatetimeIndex([VAL_CUTOFF]),
        include_target=True,
        final_dates=val_dates,
    )
    val_components = predict_components(lgb_model, ridge_model, val_frame, features, target)
    val_blend = weighted_prediction(val_components, component_weights)

    residual = oof_preds["actual"] - oof_blend
    oof_corr, val_corr, residual_alpha, residual_feat = fit_residual_model(oof_frame, residual, val_frame)
    residual_scale = tune_residual_scale(oof_preds["actual"], oof_blend, oof_corr)
    val_corrected = np.clip(val_blend + residual_scale * val_corr, 1.0, None)

    val_predictions = {**val_components, "blend": val_blend, "holiday_corrected": val_corrected}
    oof_predictions = {
        **{k: oof_preds[k] for k in COMPONENTS},
        "actual": oof_preds["actual"],
        "blend": oof_blend,
        "holiday_corrected": np.clip(oof_blend + residual_scale * oof_corr, 1.0, None),
        "residual_correction": oof_corr,
    }

    meta = {
        "target": target,
        "component_weights": component_weights,
        "residual_scale": residual_scale,
        "residual_ridge_alpha": residual_alpha,
        "n_residual_features": len(residual_feat),
        "residual_features": residual_feat,
    }
    (OUT / f"{target.lower()}_experiment_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return TargetExperiment(
        target=target,
        val_frame=val_frame,
        oof_frame=oof_frame,
        component_weights=component_weights,
        residual_scale=residual_scale,
        val_predictions=val_predictions,
        oof_predictions=oof_predictions,
        features=features,
    )


def combine_cogs_ratio(
    sales: pd.DataFrame,
    rev_exp: TargetExperiment,
    cogs_exp: TargetExperiment,
) -> tuple[float, np.ndarray, np.ndarray]:
    oof_ratio = ratio_prior_for_rows(sales, rev_exp.oof_frame)
    val_ratio = ratio_prior_for_rows(sales, cogs_exp.val_frame.assign(cutoff=VAL_CUTOFF))
    oof_cogs_from_ratio = rev_exp.oof_predictions["blend"] * oof_ratio
    val_cogs_from_ratio = rev_exp.val_predictions["blend"] * val_ratio

    actual = cogs_exp.oof_predictions["actual"]
    best = (float("inf"), 0.0)
    for w in np.arange(0.0, 1.05, 0.05):
        pred = (1.0 - w) * cogs_exp.oof_predictions["holiday_corrected"] + w * oof_cogs_from_ratio
        mae = float(mean_absolute_error(actual, pred))
        if mae < best[0]:
            best = (mae, float(w))
    w_ratio = best[1]
    val_ratio_blend = np.clip(
        (1.0 - w_ratio) * cogs_exp.val_predictions["holiday_corrected"] + w_ratio * val_cogs_from_ratio,
        1.0,
        None,
    )
    return w_ratio, val_cogs_from_ratio, val_ratio_blend


def plot_validation(actual_df: pd.DataFrame, pred_df: pd.DataFrame) -> Path:
    long = actual_df.merge(pred_df, on="Date")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for ax, target in zip(axes, TARGETS):
        ax.plot(long["Date"], long[f"{target}_actual"], label="actual", color="#111111", linewidth=1.3)
        ax.plot(long["Date"], long[f"{target}_pred"], label="advanced pred", color="#1f77b4", linewidth=1.1)
        ax.set_title(f"Validation forecast: {target}")
        ax.set_ylabel(target)
        ax.legend(loc="upper right")
    fig.tight_layout()
    path = OUT / "validation_2021_2022_advanced_forecast.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_component_metrics(metrics_df: pd.DataFrame) -> Path:
    data = metrics_df[(metrics_df["split"] == "validation_548d") & (metrics_df["metric"] == "mae")].copy()
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=data, x="model", y="value", hue="target", ax=ax)
    ax.set_title("Validation MAE by model/component")
    ax.set_xlabel("")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = OUT / "validation_mae_by_component.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    metrics_df: pd.DataFrame,
    rev_exp: TargetExperiment,
    cogs_exp: TargetExperiment,
    ratio_weight: float,
    files: dict[str, str],
) -> Path:
    report = OUT / "advanced_time_series_experiment_report.md"
    lines = [
        "# Advanced Time-Series Experiment Report",
        "",
        "## Validation Split",
        "",
        "- User-proposed split `2021-01-01..2022-07-01` has 547 days.",
        "- Kaggle test `2023-01-01..2024-07-01` has 548 days because 2024 is a leap year.",
        "- Main validation in this experiment uses `2021-01-01..2022-07-02` to match the 548-day horizon.",
        "- Forecast cutoff is `2020-12-31`; no target after that date is used as a feature.",
        "",
        "## Research Mapping",
        "",
        "- Recursive forecasting uses previous predictions as future lag inputs; this is why recursive lag7 becomes generated after the first week.",
        "- M5 winning methods support our backbone choice: model diversity with LightGBM, recursive/non-recursive variants, calendar/events, prices/promotions, and multiple train sets.",
        "- Forecast-combination literature supports combining forecasts from different perspectives to reduce reliance on a single best model.",
        "",
        "## Tested Methods",
        "",
        "1. Direct LightGBM component.",
        "2. Ridge component for smooth linear seasonality.",
        "3. Day-of-year prior.",
        "4. Safe lag730 analog component.",
        "5. Non-negative MAE-tuned blend over the four components using historical pseudo-cutoff OOF.",
        "6. Holiday/event residual correction with a regularized Ridge model over calendar/holiday/window features.",
        "7. COGS ratio correction using `Revenue_pred * historical COGS/Revenue day-of-year ratio`.",
        "",
        "## Learned Weights",
        "",
        "Revenue component weights:",
        "",
        pd.DataFrame([rev_exp.component_weights]).to_markdown(index=False, floatfmt=".2f"),
        "",
        "COGS component weights:",
        "",
        pd.DataFrame([cogs_exp.component_weights]).to_markdown(index=False, floatfmt=".2f"),
        "",
        f"Revenue holiday residual scale: `{rev_exp.residual_scale:.2f}`",
        "",
        f"COGS holiday residual scale: `{cogs_exp.residual_scale:.2f}`",
        "",
        f"COGS ratio blend weight: `{ratio_weight:.2f}`",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{k}`: `{v}`" for k, v in files.items())
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_final_candidates(
    sales: pd.DataFrame,
    rev_exp: TargetExperiment,
    cogs_exp: TargetExperiment,
    ratio_weight: float,
) -> dict[str, str]:
    """Apply the learned low-risk corrections to existing final direct/base files.

    This does not read sample target values.  It creates candidate files only;
    they still need leaderboard validation before replacing the current best.
    """
    files: dict[str, str] = {}
    direct_path = BASE_ARTIFACTS / "submission_direct_factory_regime_recovery.csv"
    base_path = BASE_ARTIFACTS / "m5b50300515.csv"
    if not direct_path.exists() or not base_path.exists():
        return files

    direct = pd.read_csv(direct_path, parse_dates=["Date"])
    base = pd.read_csv(base_path, parse_dates=["Date"])

    # Conservative event correction: only apply 50% of the validation-learned
    # residual scale to avoid overfitting a single holdout window.
    final = direct.copy()
    for exp in (rev_exp, cogs_exp):
        target = exp.target
        frame = build_feature_rows(
            sales,
            target,
            pd.DatetimeIndex([TRAIN_END]),
            include_target=False,
            final_dates=pd.date_range(FORECAST_START, FORECAST_END, freq="D"),
        )
        # Fit residual model once on the available OOF residuals.
        residual = exp.oof_predictions["actual"] - exp.oof_predictions["blend"]
        _, corr, _, _ = fit_residual_model(exp.oof_frame, residual, frame)
        final[target] = np.clip(final[target].to_numpy(dtype=float) + 0.5 * exp.residual_scale * corr, 1.0, None)

    # Ratio-correct COGS conservatively if the validation OOF selected it.
    if ratio_weight > 0:
        ratio_rows = final[["Date"]].copy()
        ratio_rows["cutoff"] = TRAIN_END
        ratio_prior = ratio_prior_for_rows(sales, ratio_rows)
        ratio_cogs = final["Revenue"].to_numpy(dtype=float) * ratio_prior
        final["COGS"] = np.clip((1.0 - 0.5 * ratio_weight) * final["COGS"].to_numpy(dtype=float) + (0.5 * ratio_weight) * ratio_cogs, 1.0, None)

    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}
    final_regime = normalise_yearly(final, levels)
    direct_corrected_path = OUT / "submission_direct_holiday_ratio_corrected_regime.csv"
    export_submission(final_regime, direct_corrected_path)
    files["direct_holiday_ratio_corrected"] = str(direct_corrected_path)

    for w in (0.10, 0.15, 0.20):
        blend = base[["Date"]].copy()
        for target in TARGETS:
            blend[target] = (1 - w) * base[target].to_numpy(dtype=float) + w * final_regime[target].to_numpy(dtype=float)
        path = OUT / f"submission_m5_advanced_direct_blend_{int((1-w)*100):02d}_{int(w*100):02d}.csv"
        export_submission(blend, path)
        files[f"m5_advanced_direct_{int((1-w)*100):02d}_{int(w*100):02d}"] = str(path)
    return files


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = load_sales()

    rev_exp = run_target(sales, "Revenue")
    cogs_exp = run_target(sales, "COGS")
    ratio_weight, cogs_from_ratio, cogs_ratio_blend = combine_cogs_ratio(sales, rev_exp, cogs_exp)

    val_actual = rev_exp.val_frame[["Date", "Revenue"]].merge(
        cogs_exp.val_frame[["Date", "COGS"]],
        on="Date",
    ).rename(columns={"Revenue": "Revenue_actual", "COGS": "COGS_actual"})
    val_pred = pd.DataFrame(
        {
            "Date": rev_exp.val_frame["Date"],
            "Revenue_pred": rev_exp.val_predictions["holiday_corrected"],
            "COGS_pred": cogs_ratio_blend,
        }
    )
    val_pred.to_csv(OUT / "validation_advanced_predictions.csv", index=False)

    rows = []
    for target, exp in (("Revenue", rev_exp), ("COGS", cogs_exp)):
        actual = exp.val_frame[target].to_numpy(dtype=float)
        for model_name, pred in exp.val_predictions.items():
            for metric_name, value in evaluate_predictions(actual, pred).items():
                rows.append(
                    {
                        "split": "validation_548d",
                        "target": target,
                        "model": model_name,
                        "metric": metric_name,
                        "value": value,
                    }
                )
    cogs_actual = cogs_exp.val_frame["COGS"].to_numpy(dtype=float)
    for model_name, pred in (("cogs_from_revenue_ratio", cogs_from_ratio), ("cogs_ratio_blend", cogs_ratio_blend)):
        for metric_name, value in evaluate_predictions(cogs_actual, pred).items():
            rows.append(
                {
                    "split": "validation_548d",
                    "target": "COGS",
                    "model": model_name,
                    "metric": metric_name,
                    "value": value,
                }
            )

    # Calendar-end 547-day view requested by the user.
    user_mask = rev_exp.val_frame["Date"].between(VAL_START, VAL_END_USER).to_numpy()
    for target, exp, pred in (
        ("Revenue", rev_exp, rev_exp.val_predictions["holiday_corrected"]),
        ("COGS", cogs_exp, cogs_ratio_blend),
    ):
        actual = exp.val_frame.loc[user_mask, target].to_numpy(dtype=float)
        pred_user = pred[user_mask]
        for metric_name, value in evaluate_predictions(actual, pred_user).items():
            rows.append(
                {
                    "split": "validation_user_547d",
                    "target": target,
                    "model": "advanced_final",
                    "metric": metric_name,
                    "value": value,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "advanced_validation_metrics.csv", index=False)
    files = {
        "validation_predictions": str(OUT / "validation_advanced_predictions.csv"),
        "metrics": str(OUT / "advanced_validation_metrics.csv"),
        "validation_plot": str(plot_validation(val_actual, val_pred)),
        "component_mae_plot": str(plot_component_metrics(metrics_df)),
    }
    files.update(build_final_candidates(sales, rev_exp, cogs_exp, ratio_weight))
    report = write_report(metrics_df, rev_exp, cogs_exp, ratio_weight, files)
    files["report"] = str(report)
    (OUT / "advanced_experiment_manifest.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
