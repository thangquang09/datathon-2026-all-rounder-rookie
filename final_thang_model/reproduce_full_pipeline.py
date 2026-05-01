"""Run the full training/inference pipeline and reproduce `submission.csv`.

Unlike `reproduce_submission.py`, this script does not merely copy an existing
final artifact. It regenerates the deterministic calendar table, runs the model
training/inference scripts in the same order as the notebook, rebuilds the final
80/20 M5-direct candidate, then exports and validates `submission.csv`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


FINAL = Path(__file__).resolve().parent
ROOT = FINAL.parent
ARTIFACTS = FINAL / "model_thang" / "artifacts"
ADVANCED = ARTIFACTS / "advanced_experiments"
FINAL_CANDIDATE = ADVANCED / "submission_m5_lgb_direct_blend_80_20.csv"
SUBMISSION = FINAL / "submission.csv"


def run_script(rel_path: str) -> None:
    script = FINAL / rel_path
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FINAL) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n>>> Running {rel_path}")
    started = time.time()
    subprocess.run([sys.executable, str(script)], cwd=str(FINAL), env=env, check=True)
    print(f"<<< Done {rel_path} in {time.time() - started:.1f}s")


def generate_calendar() -> None:
    sys.path.insert(0, str(FINAL))
    from src.calendar_vn import build_vn_event_calendar

    path = FINAL / "docs" / "vietnam_calendar_events_deterministic_2012_2024.csv"
    table = build_vn_event_calendar(2012, 2024)
    table.to_csv(path, index=False)
    print(f"Generated deterministic calendar: {path}")


def validate_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if list(df.columns) != ["Date", "Revenue", "COGS"]:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    if len(df) != 548:
        raise ValueError(f"Expected 548 rows, got {len(df)} in {path}")
    if str(df["Date"].min().date()) != "2023-01-01":
        raise ValueError(f"Unexpected start date: {df['Date'].min().date()}")
    if str(df["Date"].max().date()) != "2024-07-01":
        raise ValueError(f"Unexpected end date: {df['Date'].max().date()}")
    values = df[["Revenue", "COGS"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite Revenue/COGS values")
    if (values <= 0).any():
        raise ValueError("Submission contains non-positive Revenue/COGS values")
    return df


def reproduce(skip_visuals: bool) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ADVANCED.mkdir(parents=True, exist_ok=True)

    generate_calendar()
    run_script("model_thang/forecast_pipeline.py")
    run_script("model_thang/build_v4_regime_candidate.py")
    run_script("model_thang/build_legacy_blend_regime.py")

    shape = ARTIFACTS / "submission_shape_ensemble_recovery_upper.csv"
    shape_regime = ARTIFACTS / "submission_shape_ensemble_recovery_upper_regime_recovery.csv"
    if shape.exists() and not shape_regime.exists():
        shutil.copy2(shape, shape_regime)

    run_script("model_thang/build_m5_style_blend.py")
    run_script("model_thang/explainable_forecast_factory.py")
    run_script("model_thang/build_direct_lgb_candidates.py")
    if not skip_visuals:
        run_script("model_thang/visualize_top_features.py")

    if not FINAL_CANDIDATE.exists():
        raise FileNotFoundError(f"Missing final candidate: {FINAL_CANDIDATE}")
    validate_submission(FINAL_CANDIDATE)
    shutil.copy2(FINAL_CANDIDATE, SUBMISSION)
    validate_submission(SUBMISSION)
    print(f"\nFinal submission written: {SUBMISSION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip feature-importance PNG generation after model artifacts are built.",
    )
    args = parser.parse_args()
    reproduce(skip_visuals=args.skip_visuals)


if __name__ == "__main__":
    main()
