"""Reproduce the final submitted CSV from generated model artifacts.

This script intentionally performs the final deterministic packaging step only:
it copies the locked final candidate
`artifacts/final_candidates/submission_m5_lgb_direct_blend_80_20.csv`
to `submission.csv` and validates the result.

Run the training pipeline/notebook first when artifacts need to be regenerated.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_CANDIDATE = (
    ROOT
    / "artifacts"
    / "final_candidates"
    / "submission_m5_lgb_direct_blend_80_20.csv"
)
SUBMISSION = ROOT / "submission.csv"


def validate_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if list(df.columns) != ["Date", "Revenue", "COGS"]:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    if len(df) != 548:
        raise ValueError(f"Expected 548 rows, got {len(df)} in {path}")
    if str(df["Date"].min().date()) != "2023-01-01":
        raise ValueError(f"Unexpected start date in {path}: {df['Date'].min().date()}")
    if str(df["Date"].max().date()) != "2024-07-01":
        raise ValueError(f"Unexpected end date in {path}: {df['Date'].max().date()}")
    values = df[["Revenue", "COGS"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite Revenue/COGS values in {path}")
    if (values <= 0).any():
        raise ValueError(f"Non-positive Revenue/COGS values in {path}")
    return df


def reproduce(overwrite: bool) -> None:
    if not FINAL_CANDIDATE.exists():
        raise FileNotFoundError(
            f"Missing final candidate: {FINAL_CANDIDATE}. "
            "Run notebooks/reproduce_best_kaggle_solution.ipynb or train_save_infer_blend.py first."
        )

    validate_submission(FINAL_CANDIDATE)
    if SUBMISSION.exists() and not overwrite:
        identical = filecmp.cmp(FINAL_CANDIDATE, SUBMISSION, shallow=False)
        if not identical:
            raise FileExistsError(
                f"{SUBMISSION} already exists and differs from the final candidate. "
                "Re-run with --overwrite to replace it."
            )
    else:
        shutil.copy2(FINAL_CANDIDATE, SUBMISSION)

    validate_submission(SUBMISSION)
    if not filecmp.cmp(FINAL_CANDIDATE, SUBMISSION, shallow=False):
        raise RuntimeError("Reproduced submission is not byte-identical to the final candidate")

    print(f"candidate:  {FINAL_CANDIDATE}")
    print(f"submission: {SUBMISSION}")
    print("status: byte-identical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace submission.csv with the final candidate if it already exists.",
    )
    args = parser.parse_args()
    reproduce(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
