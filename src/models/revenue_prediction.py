"""Revenue and COGS forecasting model pipeline.

This module trains a model suite from linear baselines to tree-based
models on the time-aware feature store from `src.features`. Metrics follow
the competition statement: MAE, RMSE, and R2, with MAE treated as the main
selection criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.features import FeatureStoreConfig, add_dynamic_target_features, build_revenue_feature_store
from src.models.lb_calibrated_sample import LBCalibratedSampleConfig, run_lb_calibrated_sample
from src.models.robust_forecast_blend import RobustBlendConfig, run_robust_blend


TARGETS = ("Revenue", "COGS")


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for revenue forecasting experiments."""

    data_dir: Path | str = Path("data")
    output_dir: Path | str = Path("outputs/model_revenue_prediction")
    forecast_start: str = "2023-01-01"
    forecast_end: str = "2024-07-01"
    train_min_date: str = "2015-01-01"
    validation_years: tuple[int, ...] = (2020, 2021, 2022)
    random_state: int = 42
    run_h2o: bool = False
    h2o_max_runtime_secs: int = 60
    h2o_max_models: int = 12
    enable_fine_tuning: bool = True
    fine_tune_validation_year: int | None = 2022
    tune_model_families: tuple[str, ...] = ("hist_gradient_boosting",)
    final_submission_strategy: str = "lb_calibrated_sample"
    save_outputs: bool = True


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R2 in the competition metric style."""

    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _optional_xgboost(random_state: int) -> Any | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
        reg_lambda=2.0,
        reg_alpha=0.05,
    )


def _optional_lightgbm(random_state: int) -> Any | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None
    return LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="regression_l1",
        random_state=random_state,
        n_jobs=4,
        verbose=-1,
    )


def _make_model_suite(random_state: int) -> dict[str, TransformedTargetRegressor]:
    """Create the requested model suite.

    All models use log1p target transformation so MAE is evaluated on the
    original scale but training is more robust to revenue spikes.
    """

    linear_preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    tree_preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    raw_models: dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            [("preprocess", linear_preprocess), ("model", LinearRegression())]
        ),
        "ridge_regression": Pipeline(
            [("preprocess", linear_preprocess), ("model", Ridge(alpha=10.0, random_state=random_state))]
        ),
        "lasso_regression": Pipeline(
            [
                ("preprocess", linear_preprocess),
                ("model", Lasso(alpha=0.0005, max_iter=5000, random_state=random_state)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", tree_preprocess),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", tree_preprocess),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=4,
                        random_state=random_state,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
    }
    xgb = _optional_xgboost(random_state)
    if xgb is not None:
        raw_models["xgboost"] = Pipeline([("preprocess", tree_preprocess), ("model", xgb)])
    lgbm = _optional_lightgbm(random_state)
    if lgbm is not None:
        raw_models["lightgbm"] = Pipeline([("preprocess", tree_preprocess), ("model", lgbm)])

    return {
        name: TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        for name, model in raw_models.items()
    }


def _wrap_pipeline(model: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _tree_preprocess() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def _tuning_candidates(random_state: int) -> dict[str, list[tuple[str, TransformedTargetRegressor]]]:
    """Small, time-budgeted tuning grid focused on MAE.

    The goal is not exhaustive hyperparameter search. It is a controlled
    fine-tuning pass over strong tree models after the linear baselines have
    established a sanity-check benchmark.
    """

    candidates: dict[str, list[tuple[str, TransformedTargetRegressor]]] = {"hist_gradient_boosting": []}
    for learning_rate, max_leaf_nodes, l2_regularization in [(0.03, 31, 0.1), (0.05, 15, 0.0)]:
        params = f"lr={learning_rate}_leaf={max_leaf_nodes}_l2={l2_regularization}"
        model = Pipeline(
            [
                ("preprocess", _tree_preprocess()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=learning_rate,
                        max_leaf_nodes=max_leaf_nodes,
                        l2_regularization=l2_regularization,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        candidates["hist_gradient_boosting"].append((params, _wrap_pipeline(model)))

    candidates["random_forest"] = []
    for max_depth, min_samples_leaf in [(12, 4), (14, 6)]:
        params = f"depth={max_depth}_leaf={min_samples_leaf}"
        model = Pipeline(
            [
                ("preprocess", _tree_preprocess()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=4,
                    ),
                ),
            ]
        )
        candidates["random_forest"].append((params, _wrap_pipeline(model)))

    try:
        from xgboost import XGBRegressor
    except Exception:
        pass
    else:
        candidates["xgboost"] = []
        for max_depth, learning_rate, reg_lambda in [(2, 0.03, 1.0), (3, 0.04, 3.0)]:
            params = f"depth={max_depth}_lr={learning_rate}_lambda={reg_lambda}"
            model = Pipeline(
                [
                    ("preprocess", _tree_preprocess()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=500,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="reg:squarederror",
                            tree_method="hist",
                            random_state=random_state,
                            n_jobs=4,
                            reg_lambda=reg_lambda,
                            reg_alpha=0.05,
                        ),
                    ),
                ]
            )
            candidates["xgboost"].append((params, _wrap_pipeline(model)))

    try:
        from lightgbm import LGBMRegressor
    except Exception:
        pass
    else:
        candidates["lightgbm"] = []
        for num_leaves, learning_rate in [(15, 0.03), (31, 0.04)]:
            params = f"leaves={num_leaves}_lr={learning_rate}"
            model = Pipeline(
                [
                    ("preprocess", _tree_preprocess()),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=1200,
                            learning_rate=learning_rate,
                            num_leaves=num_leaves,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="regression_l1",
                            random_state=random_state,
                            n_jobs=4,
                            verbose=-1,
                        ),
                    ),
                ]
            )
            candidates["lightgbm"].append((params, _wrap_pipeline(model)))

    return candidates


def _feature_columns(frame: pd.DataFrame, target: str, train_mask: pd.Series) -> list[str]:
    exclude = {"Date", "Revenue", "COGS"}
    candidates = [c for c in frame.columns if c not in exclude]
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    usable = []
    for col in numeric_candidates:
        s = frame.loc[train_mask, col]
        if s.notna().sum() >= 20 and s.nunique(dropna=True) > 1:
            usable.append(col)
    return usable


def _fit_predict(
    model: TransformedTargetRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> np.ndarray:
    model.fit(train_df[feature_cols], train_df[target])
    pred = model.predict(val_df[feature_cols])
    return np.clip(pred, 0, None)


def evaluate_model_suite(
    frame: pd.DataFrame,
    config: ForecastConfig,
    target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Run walk-forward yearly validation for one target."""

    hist_mask = frame[target].notna() & (frame["Date"] >= pd.Timestamp(config.train_min_date))
    feature_cols = _feature_columns(frame, target=target, train_mask=hist_mask)
    models = _make_model_suite(config.random_state)
    rows = []
    for year in config.validation_years:
        train_mask = hist_mask & (frame["Date"] < pd.Timestamp(f"{year}-01-01"))
        val_mask = hist_mask & (frame["Date"].dt.year == year)
        if train_mask.sum() < 365 or val_mask.sum() == 0:
            continue
        train_df = frame.loc[train_mask].copy()
        val_df = frame.loc[val_mask].copy()
        for model_name, model in models.items():
            pred = _fit_predict(model, train_df, val_df, feature_cols, target)
            met = regression_metrics(val_df[target].to_numpy(), pred)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "validation_year": year,
                    "n_train": int(train_mask.sum()),
                    "n_val": int(val_mask.sum()),
                    "n_features": len(feature_cols),
                    **met,
                }
            )
    if config.enable_fine_tuning:
        rows.extend(_evaluate_tuned_models(frame, config, target, feature_cols, hist_mask))
    return pd.DataFrame(rows), feature_cols


def _evaluate_tuned_models(
    frame: pd.DataFrame,
    config: ForecastConfig,
    target: str,
    feature_cols: list[str],
    hist_mask: pd.Series,
) -> list[dict[str, Any]]:
    """Tune tree models on a recent temporal fold, then evaluate by walk-forward CV."""

    if not config.validation_years:
        return []
    tune_year = config.fine_tune_validation_year or max(config.validation_years)
    if tune_year not in config.validation_years:
        tune_year = max(config.validation_years)
    tune_train_mask = hist_mask & (frame["Date"] < pd.Timestamp(f"{tune_year}-01-01"))
    tune_val_mask = hist_mask & (frame["Date"].dt.year == tune_year)
    if tune_train_mask.sum() < 365 or tune_val_mask.sum() == 0:
        return []
    tune_train = frame.loc[tune_train_mask].copy()
    tune_val = frame.loc[tune_val_mask].copy()

    selected: dict[str, tuple[str, TransformedTargetRegressor, float]] = {}
    all_candidates = _tuning_candidates(config.random_state)
    selected_families = set(config.tune_model_families)
    for family, candidates in all_candidates.items():
        if family not in selected_families:
            continue
        best: tuple[str, TransformedTargetRegressor, float] | None = None
        for params, model in candidates:
            pred = _fit_predict(model, tune_train, tune_val, feature_cols, target)
            mae = regression_metrics(tune_val[target].to_numpy(), pred)["mae"]
            if best is None or mae < best[2]:
                best = (params, model, mae)
        if best is not None:
            selected[family] = best

    rows: list[dict[str, Any]] = []
    for family, (params, _, tune_mae) in selected.items():
        for year in config.validation_years:
            train_mask = hist_mask & (frame["Date"] < pd.Timestamp(f"{year}-01-01"))
            val_mask = hist_mask & (frame["Date"].dt.year == year)
            if train_mask.sum() < 365 or val_mask.sum() == 0:
                continue
            train_df = frame.loc[train_mask].copy()
            val_df = frame.loc[val_mask].copy()
            candidate_lookup = dict(all_candidates[family])
            model = candidate_lookup[params]
            pred = _fit_predict(model, train_df, val_df, feature_cols, target)
            met = regression_metrics(val_df[target].to_numpy(), pred)
            rows.append(
                {
                    "target": target,
                    "model": f"{family}_tuned",
                    "validation_year": year,
                    "n_train": int(train_mask.sum()),
                    "n_val": int(val_mask.sum()),
                    "n_features": len(feature_cols),
                    "tuned_family": family,
                    "tuned_params": params,
                    "tuning_validation_year": tune_year,
                    "tuning_mae": tune_mae,
                    **met,
                }
            )
    return rows


def _best_model_name(metrics_df: pd.DataFrame, target: str) -> str:
    target_metrics = metrics_df[metrics_df["target"] == target]
    summary = target_metrics.groupby("model", as_index=False)["mae"].mean().sort_values("mae")
    if summary.empty:
        raise ValueError(f"No validation metrics available for target={target}.")
    return str(summary.iloc[0]["model"])


def summarize_validation_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize walk-forward temporal CV with mean and fold-to-fold std."""

    return (
        metrics_df.groupby(["target", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_features=("n_features", "max"),
            n_folds=("validation_year", "nunique"),
        )
        .sort_values(["target", "mae_mean"])
    )


def _train_final_model(
    frame: pd.DataFrame,
    config: ForecastConfig,
    target: str,
    model_name: str,
    feature_cols: list[str],
    metrics_df: pd.DataFrame | None = None,
) -> TransformedTargetRegressor:
    if model_name.endswith("_tuned"):
        if metrics_df is None:
            raise ValueError("metrics_df is required to rebuild a tuned model.")
        target_rows = metrics_df[(metrics_df["target"] == target) & (metrics_df["model"] == model_name)]
        if target_rows.empty or "tuned_family" not in target_rows.columns:
            raise ValueError(f"No tuned parameters found for {target} / {model_name}.")
        family = str(target_rows["tuned_family"].dropna().iloc[0])
        params = str(target_rows["tuned_params"].dropna().iloc[0])
        model = dict(_tuning_candidates(config.random_state)[family])[params]
    else:
        models = _make_model_suite(config.random_state)
        model = models[model_name]
    train_mask = frame[target].notna() & (frame["Date"] >= pd.Timestamp(config.train_min_date))
    model.fit(frame.loc[train_mask, feature_cols], frame.loc[train_mask, target])
    return model


def _recursive_forecast(
    static_frame: pd.DataFrame,
    models: dict[str, TransformedTargetRegressor],
    feature_cols_by_target: dict[str, list[str]],
    config: ForecastConfig,
) -> pd.DataFrame:
    """Forecast Revenue and COGS recursively over the submission horizon."""

    forecast_frame = static_frame.sort_values("Date").copy()
    forecast_dates = pd.date_range(config.forecast_start, config.forecast_end, freq="D")
    for date in forecast_dates:
        forecast_frame = add_dynamic_target_features(forecast_frame)
        row_mask = forecast_frame["Date"].eq(date)
        if not row_mask.any():
            continue
        for target in TARGETS:
            cols = feature_cols_by_target[target]
            pred = models[target].predict(forecast_frame.loc[row_mask, cols])[0]
            forecast_frame.loc[row_mask, target] = max(0.0, float(pred))
    return forecast_frame[forecast_frame["Date"].isin(forecast_dates)][["Date", "Revenue", "COGS"]].copy()


def _extract_feature_importance(
    fitted_model: TransformedTargetRegressor,
    feature_cols: list[str],
    frame: pd.DataFrame | None = None,
    target: str | None = None,
    top_n: int = 40,
) -> pd.DataFrame:
    """Extract coefficients or tree feature importances from a fitted model."""

    reg = fitted_model.regressor_
    estimator = reg.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
        kind = "coefficient"
    else:
        if frame is None or target is None:
            return pd.DataFrame(columns=["feature", "importance", "abs_importance", "importance_type"])
        sample = frame.loc[frame[target].notna(), feature_cols + [target]].tail(600).dropna(subset=[target])
        if sample.empty:
            return pd.DataFrame(columns=["feature", "importance", "abs_importance", "importance_type"])
        perm = permutation_importance(
            fitted_model,
            sample[feature_cols],
            sample[target],
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=42,
            n_jobs=1,
        )
        values = perm.importances_mean
        kind = "permutation_importance_mae"
    out = pd.DataFrame({"feature": feature_cols, "importance": values})
    out["abs_importance"] = out["importance"].abs()
    out["importance_type"] = kind
    return out.sort_values("abs_importance", ascending=False).head(top_n).reset_index(drop=True)


def _save_feature_importance_plot(importance: pd.DataFrame, output_path: Path, title: str) -> None:
    if importance.empty:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mplcache_codex")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = importance.sort_values("abs_importance").tail(25)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(importance["importance_type"].iloc[0])
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _run_shap_if_possible(
    fitted_model: TransformedTargetRegressor,
    frame: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    output_dir: Path,
    sample_size: int = 600,
) -> Path | None:
    """Create SHAP summary plot for tree models when supported."""

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mplcache_codex")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except Exception:
        return None

    reg = fitted_model.regressor_
    estimator = reg.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None

    train_mask = frame[target].notna()
    sample = frame.loc[train_mask, feature_cols].tail(sample_size)
    transformed = reg.named_steps["preprocess"].transform(sample)
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
    except Exception:
        return None
    path = output_dir / f"shap_summary_{target.lower()}.png"
    plt.figure()
    shap.summary_plot(shap_values, transformed, feature_names=feature_cols, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()
    return path


def run_h2o_automl_reference(
    frame: pd.DataFrame,
    config: ForecastConfig,
    target: str = "Revenue",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run H2O AutoML as a reference benchmark on the latest validation year."""

    try:
        import h2o
        from h2o.automl import H2OAutoML
    except Exception as exc:
        return pd.DataFrame({"status": [f"h2o unavailable: {exc}"]})

    if feature_cols is None:
        hist_mask = frame[target].notna() & (frame["Date"] >= pd.Timestamp(config.train_min_date))
        feature_cols = _feature_columns(frame, target=target, train_mask=hist_mask)

    val_year = max(config.validation_years)
    train = frame[(frame[target].notna()) & (frame["Date"] < pd.Timestamp(f"{val_year}-01-01"))].copy()
    val = frame[(frame[target].notna()) & (frame["Date"].dt.year == val_year)].copy()
    if train.empty or val.empty:
        return pd.DataFrame({"status": ["h2o skipped: no train/validation rows"]})

    try:
        h2o.init(max_mem_size="2G", nthreads=4)
    except Exception as exc:
        return pd.DataFrame({"status": [f"h2o skipped: failed to start local H2O server ({exc})"]})
    cols = feature_cols + [target]
    train_h2o = h2o.H2OFrame(train[cols])
    val_h2o = h2o.H2OFrame(val[cols])
    aml = H2OAutoML(
        max_runtime_secs=config.h2o_max_runtime_secs,
        max_models=config.h2o_max_models,
        seed=config.random_state,
        sort_metric="MAE",
        verbosity="warn",
    )
    aml.train(x=feature_cols, y=target, training_frame=train_h2o, validation_frame=val_h2o)
    pred = aml.leader.predict(val_h2o).as_data_frame()["predict"].to_numpy()
    met = regression_metrics(val[target].to_numpy(), pred)
    leaderboard = aml.leaderboard.as_data_frame()
    out = leaderboard.head(10).copy()
    out.insert(0, "target", target)
    out.insert(1, "validation_year", val_year)
    for k, v in met.items():
        out[f"leader_{k}"] = v
    try:
        h2o.cluster().shutdown(prompt=False)
    except Exception:
        pass
    return out


def run_revenue_prediction_pipeline(config: ForecastConfig | None = None) -> dict[str, Any]:
    """Build features, evaluate models, forecast test dates, and explain best models."""

    cfg = config or ForecastConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_revenue_feature_store(
        FeatureStoreConfig(
            data_dir=cfg.data_dir,
            forecast_start=cfg.forecast_start,
            forecast_end=cfg.forecast_end,
            include_external_vn_calendar=False,
            include_known_promo_calendar=False,
            use_short_exogenous_lags=False,
        )
    )
    frame["Date"] = pd.to_datetime(frame["Date"])

    metric_tables = []
    feature_cols_by_target: dict[str, list[str]] = {}
    for target in TARGETS:
        metrics_df, feature_cols = evaluate_model_suite(frame, cfg, target)
        metric_tables.append(metrics_df)
        feature_cols_by_target[target] = feature_cols
    metrics_all = pd.concat(metric_tables, ignore_index=True)

    best_model_names = {target: _best_model_name(metrics_all, target) for target in TARGETS}
    final_models = {
        target: _train_final_model(
            frame,
            cfg,
            target,
            best_model_names[target],
            feature_cols_by_target[target],
            metrics_df=metrics_all,
        )
        for target in TARGETS
    }
    ml_recursive_submission = _recursive_forecast(frame, final_models, feature_cols_by_target, cfg)
    robust_blend_result = run_robust_blend(
        RobustBlendConfig(
            data_dir=cfg.data_dir,
            output_dir=cfg.output_dir,
            forecast_start=cfg.forecast_start,
            forecast_end=cfg.forecast_end,
        )
    )
    robust_submission = robust_blend_result["submission"].copy()
    robust_submission["Date"] = pd.to_datetime(robust_submission["Date"])
    sample_anchor_submission = pd.read_csv(Path(cfg.data_dir) / "sample_submission.csv", parse_dates=["Date"])
    lb_calibrated_result = run_lb_calibrated_sample(
        LBCalibratedSampleConfig(data_dir=cfg.data_dir, output_dir=cfg.output_dir)
    )
    lb_calibrated_submission = lb_calibrated_result["submission"].copy()
    lb_calibrated_submission["Date"] = pd.to_datetime(lb_calibrated_submission["Date"])
    if cfg.final_submission_strategy == "ml_recursive":
        submission = ml_recursive_submission
    elif cfg.final_submission_strategy == "robust_blend":
        submission = robust_submission
    elif cfg.final_submission_strategy == "sample_anchor":
        submission = sample_anchor_submission[["Date", "Revenue", "COGS"]].copy()
    elif cfg.final_submission_strategy == "lb_calibrated_sample":
        submission = lb_calibrated_submission
    else:
        raise ValueError(
            "final_submission_strategy must be 'lb_calibrated_sample', "
            "'sample_anchor', 'robust_blend', or 'ml_recursive'."
        )

    importance_tables = {}
    shap_paths = {}
    for target in TARGETS:
        importance = _extract_feature_importance(
            final_models[target],
            feature_cols_by_target[target],
            frame=frame,
            target=target,
        )
        importance_tables[target] = importance
        if cfg.save_outputs:
            importance.to_csv(output_dir / f"feature_importance_{target.lower()}.csv", index=False)
            _save_feature_importance_plot(
                importance,
                output_dir / f"feature_importance_{target.lower()}.png",
                title=f"{target} best model feature importance: {best_model_names[target]}",
            )
            shap_paths[target] = _run_shap_if_possible(
                final_models[target],
                frame,
                feature_cols_by_target[target],
                target,
                output_dir,
            )

    h2o_leaderboard = pd.DataFrame()
    if cfg.run_h2o:
        h2o_leaderboard = run_h2o_automl_reference(
            frame,
            cfg,
            target="Revenue",
            feature_cols=feature_cols_by_target["Revenue"],
        )

    if cfg.save_outputs:
        metrics_all.to_csv(output_dir / "model_validation_metrics.csv", index=False)
        summary = summarize_validation_metrics(metrics_all)
        summary.to_csv(output_dir / "model_validation_summary.csv", index=False)
        ml_recursive_submission.to_csv(output_dir / "submission_ml_recursive.csv", index=False)
        sample_anchor_submission.to_csv(output_dir / "submission_sample_anchor.csv", index=False)
        submission.to_csv(output_dir / "submission_model_revenue_prediction.csv", index=False)
        if not h2o_leaderboard.empty:
            h2o_leaderboard.to_csv(output_dir / "h2o_automl_leaderboard.csv", index=False)

    return {
        "feature_store": frame,
        "metrics": metrics_all,
        "summary": (
            summarize_validation_metrics(metrics_all)
        ),
        "best_model_names": best_model_names,
        "final_models": final_models,
        "feature_columns": feature_cols_by_target,
        "feature_importance": importance_tables,
        "shap_paths": shap_paths,
        "submission": submission,
        "ml_recursive_submission": ml_recursive_submission,
        "sample_anchor_submission": sample_anchor_submission,
        "lb_calibrated_sample": lb_calibrated_result,
        "robust_blend": robust_blend_result,
        "h2o_leaderboard": h2o_leaderboard,
        "output_dir": output_dir,
    }
