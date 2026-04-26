"""Log every submission and local CV result to MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"
LEDGER_FILE = ROOT / "outputs" / "submission_ledger.csv"


def mlflow_setup(experiment_name: str = "datathon-2026-round-1") -> None:
    import mlflow

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(experiment_name)


def log_submission(
    name: str,
    strategy: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    csv_path: Path,
    public_score: float | None = None,
) -> None:
    import mlflow

    mlflow_setup()
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("strategy", strategy)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, float(v))
        if public_score is not None:
            mlflow.log_metric("public_score", float(public_score))
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), artifact_path="submissions")

    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {"name": name, "strategy": strategy, "csv": str(csv_path), "public_score": public_score, **params, **metrics}
    if LEDGER_FILE.exists():
        ledger = pd.read_csv(LEDGER_FILE)
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    else:
        ledger = pd.DataFrame([row])
    ledger.to_csv(LEDGER_FILE, index=False)


def backfill_from_history() -> None:
    """Seed the ledger with observed Kaggle submissions so far."""
    observed = [
        # (file, strategy, params, public_score)
        ("sub_1p33_1p39_cog_1p38_1p50.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.39, "cog_2023": 1.38, "cog_2024": 1.50}, 697744.14),
        ("sub_1p33_1p40_cog_1p38_1p50.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.40, "cog_2023": 1.38, "cog_2024": 1.50}, 697902.42),
        ("sub_cog_1p40_1p52.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.42, "cog_2023": 1.40, "cog_2024": 1.52}, 700403.63),
        ("sub_cog_1p42_1p55.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.42, "cog_2023": 1.42, "cog_2024": 1.55}, 704514.46),
        ("sub_1p33_1p42_cog_high.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.42, "cog_2023": 1.38, "cog_2024": 1.50}, 698659.50),
        ("sub_1p33_1p42_cog_low.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.42, "cog_2023": 1.30, "cog_2024": 1.40}, 707410.27),
        ("sub_1p32_1p41.csv", "sample_scaled", {"rev_2023": 1.32, "rev_2024": 1.41, "cog_2023": 1.32, "cog_2024": 1.41}, 700785.33),
        ("sub_scale_1p34_1p44.csv", "sample_scaled", {"rev_2023": 1.34, "rev_2024": 1.44, "cog_2023": 1.34, "cog_2024": 1.44}, 702624.71),
        ("sub_scale_1p33_1p42.csv", "sample_scaled", {"rev_2023": 1.33, "rev_2024": 1.42, "cog_2023": 1.33, "cog_2024": 1.42}, 700470.22),
        ("sub_scale_1p40_1p55.csv", "sample_scaled", {"rev_2023": 1.40, "rev_2024": 1.55, "cog_2023": 1.40, "cog_2024": 1.55}, 741524.04),
        ("sub_scale_1p35_1p45.csv", "sample_scaled", {"rev_2023": 1.35, "rev_2024": 1.45, "cog_2023": 1.35, "cog_2024": 1.45}, 703488.54),
        ("sub_scale_1p29_1p39.csv", "sample_scaled", {"rev_2023": 1.29, "rev_2024": 1.39, "cog_2023": 1.29, "cog_2024": 1.39}, 705526.43),
        ("sub_scale_1p20.csv", "sample_scaled", {"rev_2023": 1.20, "rev_2024": 1.20, "cog_2023": 1.20, "cog_2024": 1.20}, 817912.48),
        ("sub_scale_1p00.csv", "sample_scaled", {"rev_2023": 1.00, "rev_2024": 1.00, "cog_2023": 1.00, "cog_2024": 1.00}, 1225931.14),
        ("sub_shape100.csv", "seasonal_blend", {"sample_shape_weight": 1.00}, 704975.09),
        ("sub_shape80.csv", "seasonal_blend", {"sample_shape_weight": 0.80}, 706906.25),
        ("sub_shape60.csv", "seasonal_blend", {"sample_shape_weight": 0.60}, 724732.80),
        ("sub_best_sample_shape45.csv", "seasonal_blend", {"sample_shape_weight": 0.45}, 754161.12),
        ("sub_dow_blend_v1.csv", "custom_shape", {"shape": "doy_dow_blend", "rev_2023": 1.37, "rev_2024": 1.38}, 805123.44),
        ("sub_blend50.csv", "mix_sample_custom", {"sample_w": 0.5}, 752006.63),
        ("sub_best_sample_shape30.csv", "seasonal_blend", {"sample_shape_weight": 0.30}, 792349.33),
    ]
    import mlflow
    mlflow_setup()
    for fname, strat, params, score in observed:
        with mlflow.start_run(run_name=fname):
            mlflow.set_tag("strategy", strat)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("public_score", score)
            path = ROOT / "outputs" / "candidates" / fname
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="submissions")


if __name__ == "__main__":
    backfill_from_history()
    print(f"Backfill complete. MLflow dir: {MLRUNS_DIR}")
