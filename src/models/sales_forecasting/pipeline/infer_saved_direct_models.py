"""Run inference from saved direct-factory models.

This script loads the LightGBM and Ridge models saved by
`explainable_forecast_factory.py`, rebuilds deterministic future feature rows,
and writes direct inference outputs. It does not train models.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sales_forecasting import ARTIFACTS_DIR

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sales_forecasting.pipeline.explainable_forecast_factory import (  # noqa: E402
    MODEL_OUT,
    build_feature_rows,
    predict_lgb,
    predict_ridge,
)
from src.models.sales_forecasting.pipeline.forecast_pipeline import (  # noqa: E402
    FORECAST_END,
    FORECAST_START,
    TARGETS,
    TRAIN_END,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)

OUT = ARTIFACTS_DIR
INFERENCE_OUT = OUT / "inference"


def load_saved_target(target: str) -> dict:
    import lightgbm as lgb

    target_dir = MODEL_OUT / target
    metadata_path = target_dir / "metadata.json"
    lgb_path = target_dir / "lightgbm.txt"
    ridge_path = target_dir / "ridge_pipeline.pkl"
    if not metadata_path.exists() or not lgb_path.exists() or not ridge_path.exists():
        raise FileNotFoundError(
            f"Missing saved model files for {target} under {target_dir}. "
            "Run pipeline/explainable_forecast_factory.py first."
        )
    metadata = json.loads(metadata_path.read_text())
    lgb_model = lgb.Booster(model_file=str(lgb_path))
    with ridge_path.open("rb") as f:
        ridge_model = pickle.load(f)
    return {
        "metadata": metadata,
        "lgb_model": lgb_model,
        "ridge_model": ridge_model,
    }


def predict_saved_direct_component(sales: pd.DataFrame, target: str) -> pd.DataFrame:
    saved = load_saved_target(target)
    metadata = saved["metadata"]
    feature_cols = metadata["feature_cols"]
    weights = metadata["weights"]

    dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    frame = build_feature_rows(
        sales,
        target,
        pd.DatetimeIndex([TRAIN_END]),
        include_target=False,
        final_dates=dates,
    )
    pred_lgb = predict_lgb(saved["lgb_model"], frame, feature_cols)
    pred_ridge = predict_ridge(saved["ridge_model"], frame, feature_cols)
    pred_doy = frame[f"{target}_doy_mean_cutoff"].to_numpy(dtype=float)
    pred = (
        weights.get("lgb", 0.0) * pred_lgb
        + weights.get("ridge", 0.0) * pred_ridge
        + weights.get("doy_prior", 0.0) * pred_doy
    )
    return pd.DataFrame(
        {
            "Date": dates,
            target: pred.clip(min=1.0),
            f"{target}_lgb": pred_lgb,
            f"{target}_ridge": pred_ridge,
            f"{target}_doy_prior": pred_doy,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUT.mkdir(parents=True, exist_ok=True)
    sales = load_sales()
    dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")

    direct = pd.DataFrame({"Date": dates})
    component_debug = pd.DataFrame({"Date": dates})
    for target in TARGETS:
        part = predict_saved_direct_component(sales, target)
        direct[target] = part[target].to_numpy()
        for col in part.columns:
            if col != "Date" and col != target:
                component_debug[col] = part[col].to_numpy()

    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}
    direct_regime = normalise_yearly(direct, levels)

    raw_path = INFERENCE_OUT / "submission_direct_factory_raw_from_saved_models.csv"
    regime_path = INFERENCE_OUT / "submission_direct_factory_regime_from_saved_models.csv"
    debug_path = INFERENCE_OUT / "direct_factory_component_debug_from_saved_models.csv"
    export_submission(direct, raw_path)
    export_submission(direct_regime, regime_path)
    component_debug.to_csv(debug_path, index=False)

    # Keep canonical artifact names in sync for downstream blend scripts.
    export_submission(direct, OUT / "submission_direct_factory_raw.csv")
    export_submission(direct_regime, OUT / "submission_direct_factory_regime_recovery.csv")
    component_debug.to_csv(OUT / "direct_factory_component_debug.csv", index=False)

    print(json.dumps({
        "raw": str(raw_path),
        "regime": str(regime_path),
        "component_debug": str(debug_path),
    }, indent=2))


if __name__ == "__main__":
    main()
