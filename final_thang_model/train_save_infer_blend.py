"""Train models, save them, run saved-model inference, and export submission.

This is the end-to-end Python entrypoint for the final package:

1. generate the deterministic Vietnam calendar CSV from Gregorian dates;
2. train all component pipelines;
3. save direct LightGBM/Ridge models;
4. run direct inference by loading the saved models;
5. blend M5-style base inference with direct-model inference;
6. write `submission.csv`.

It is intentionally equivalent to the notebook flow but executable as a script.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ARTIFACTS = ROOT / "model_thang" / "artifacts"
ADV = ARTIFACTS / "advanced_experiments"
INFERENCE = ARTIFACTS / "inference"
MODELS = ARTIFACTS / "saved_models"
FINAL_CANDIDATE = ADV / "submission_m5_lgb_direct_blend_80_20.csv"
SUBMISSION = ROOT / "submission.csv"


def run_script(rel_path: str) -> None:
    script = ROOT / rel_path
    if not script.exists():
        raise FileNotFoundError(script)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n>>> {rel_path}")
    started = time.time()
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, check=True)
    print(f"<<< {rel_path} done in {time.time() - started:.1f}s")


def generate_calendar() -> Path:
    sys.path.insert(0, str(ROOT))
    from src.calendar_vn import build_vn_event_calendar

    path = ROOT / "docs" / "vietnam_calendar_events_deterministic_2012_2024.csv"
    df = build_vn_event_calendar(2012, 2024)
    df.to_csv(path, index=False)
    print(f"generated calendar: {path}")
    return path


def ensure_shape_regime_alias() -> None:
    shape = ARTIFACTS / "submission_shape_ensemble_recovery_upper.csv"
    shape_regime = ARTIFACTS / "submission_shape_ensemble_recovery_upper_regime_recovery.csv"
    if shape.exists() and not shape_regime.exists():
        shutil.copy2(shape, shape_regime)


def validate_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if list(df.columns) != ["Date", "Revenue", "COGS"]:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != 548:
        raise ValueError(f"Expected 548 rows, got {len(df)}")
    if str(df["Date"].min().date()) != "2023-01-01":
        raise ValueError(f"Unexpected start date: {df['Date'].min().date()}")
    if str(df["Date"].max().date()) != "2024-07-01":
        raise ValueError(f"Unexpected end date: {df['Date'].max().date()}")
    values = df[["Revenue", "COGS"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite values")
    if (values <= 0).any():
        raise ValueError("Submission contains non-positive values")
    return df


def export_final_submission() -> None:
    if not FINAL_CANDIDATE.exists():
        raise FileNotFoundError(FINAL_CANDIDATE)
    validate_submission(FINAL_CANDIDATE)
    shutil.copy2(FINAL_CANDIDATE, SUBMISSION)
    validate_submission(SUBMISSION)
    print(f"exported final submission: {SUBMISSION}")


def write_reproduce_manifest(calendar_path: Path) -> None:
    manifest = {
        "calendar": str(calendar_path),
        "models_dir": str(MODELS),
        "inference_dir": str(INFERENCE),
        "final_candidate": str(FINAL_CANDIDATE),
        "submission": str(SUBMISSION),
        "pipeline": [
            "model_thang/forecast_pipeline.py",
            "model_thang/build_v4_regime_candidate.py",
            "model_thang/build_legacy_blend_regime.py",
            "model_thang/build_m5_style_blend.py",
            "model_thang/explainable_forecast_factory.py",
            "model_thang/infer_saved_direct_models.py",
            "model_thang/build_direct_lgb_candidates.py",
            "model_thang/visualize_top_features.py",
        ],
    }
    (ARTIFACTS / "train_save_infer_blend_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-visuals", action="store_true", help="Skip feature-importance PNG generation.")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ADV.mkdir(parents=True, exist_ok=True)
    INFERENCE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    calendar_path = generate_calendar()
    run_script("model_thang/forecast_pipeline.py")
    run_script("model_thang/build_v4_regime_candidate.py")
    run_script("model_thang/build_legacy_blend_regime.py")
    ensure_shape_regime_alias()
    run_script("model_thang/build_m5_style_blend.py")
    run_script("model_thang/explainable_forecast_factory.py")
    run_script("model_thang/infer_saved_direct_models.py")
    run_script("model_thang/build_direct_lgb_candidates.py")
    if not args.skip_visuals:
        run_script("model_thang/visualize_top_features.py")
    export_final_submission()
    write_reproduce_manifest(calendar_path)


if __name__ == "__main__":
    main()
