"""Build a strong v4-feature candidate with train-only regime recovery levels.

This wrapper reuses the repository's existing v4 feature builder/model runner
(`src.final_model_v4`) because it contains the most complete engineered feature
set.  Unlike the original v4 runner, this script does not use LB/public level
constants and does not read sample_submission values.  It calibrates yearly
means with `forecast_pipeline.yearly_level_targets(..., "regime_recovery")`,
which is derived only from sales.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_thang.forecast_pipeline import (
    FORECAST_END,
    FORECAST_START,
    TARGETS,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)
from src.final_model_v4 import fit_and_forecast


OUT = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = load_sales()
    dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    sub = pd.DataFrame({"Date": dates})
    raw_means = {}

    for target in TARGETS:
        raw, _, _, _ = fit_and_forecast(target)
        sub[target] = raw
        raw_means[target] = {
            str(year): float(sub.loc[sub["Date"].dt.year == year, target].mean())
            for year in (2023, 2024)
        }

    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}
    calibrated = normalise_yearly(sub, levels)

    raw_path = OUT / "submission_v4_raw.csv"
    final_path = OUT / "submission_v4_regime_recovery.csv"
    export_submission(sub, raw_path)
    export_submission(calibrated, final_path)

    summary = {
        "raw_file": str(raw_path),
        "regime_recovery_file": str(final_path),
        "raw_means": raw_means,
        "levels": levels,
        "leakage_policy": [
            "src.final_model_v4 builds features from train CSVs only",
            "this wrapper does not read sample_submission",
            "yearly calibration uses sales.csv 2014-2018 and 2022 recovery only",
        ],
    }
    with open(OUT / "v4_regime_candidate.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
