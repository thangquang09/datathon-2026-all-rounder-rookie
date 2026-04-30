"""Build an M5-style model-diversity blend from no-leak component forecasts.

This script does not read `sample_submission.csv`.  It consumes component
forecast files already generated from train CSVs, normalises each component to
train-derived regime-recovery yearly levels, and exports several low-dimensional
weighted averages plus diagnostics.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_thang.forecast_pipeline import (
    TARGETS,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"


COMPONENTS = {
    "v1": OUT / "legacy_v1_regime_recovery.csv",
    "v2": OUT / "legacy_v2_regime_recovery.csv",
    "v3": OUT / "legacy_v3_regime_recovery.csv",
    "v4": OUT / "legacy_v4_regime_recovery.csv",
    "standalone": OUT / "submission_model_regime_recovery.csv",
    "shape": OUT / "submission_shape_ensemble_recovery_upper_regime_recovery.csv",
}

OPTIONAL_RAW_COMPONENTS = {
    "v5_huber": ROOT / "outputs" / "final_v5" / "model_v5_raw.csv",
}

BLENDS = {
    "m5blend_50_30_05_15.csv": {"v1": 0.50, "v2": 0.30, "v3": 0.05, "v4": 0.15},
    "m5blend_45_30_05_15_05.csv": {
        "v1": 0.45,
        "v2": 0.30,
        "v3": 0.05,
        "v4": 0.15,
        "standalone": 0.05,
    },
    "m5blend_50_25_00_15_10.csv": {
        "v1": 0.50,
        "v2": 0.25,
        "v3": 0.00,
        "v4": 0.15,
        "shape": 0.10,
    },
}


def read_component(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if list(df.columns) != ["Date", "Revenue", "COGS"]:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    if len(df) != 548:
        raise ValueError(f"Unexpected row count in {path}: {len(df)}")
    if not np.isfinite(df[list(TARGETS)].to_numpy()).all():
        raise ValueError(f"Non-finite forecast values in {path}")
    if (df[list(TARGETS)] <= 0).any().any():
        raise ValueError(f"Non-positive forecast values in {path}")
    return df


def available_components(levels: dict[str, dict[int, float]]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in COMPONENTS.items():
        if path.exists():
            dfs[name] = read_component(path)

    for name, path in OPTIONAL_RAW_COMPONENTS.items():
        if path.exists():
            raw = read_component(path)
            dfs[name] = normalise_yearly(raw, levels)

    missing = sorted(name for name, path in COMPONENTS.items() if not path.exists())
    if missing:
        raise FileNotFoundError(f"Missing required components: {missing}")
    return dfs


def blend(dfs: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    unknown = sorted(set(weights) - set(dfs))
    if unknown:
        raise ValueError(f"Blend references missing components: {unknown}")
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        weights = {k: v / total for k, v in weights.items()}

    out = pd.DataFrame({"Date": next(iter(dfs.values()))["Date"]})
    for target in TARGETS:
        out[target] = sum(weights[name] * dfs[name][target].to_numpy() for name in weights)
    return out


def diagnostics(dfs: dict[str, pd.DataFrame]) -> dict:
    rows = []
    for name, df in dfs.items():
        years = df["Date"].dt.year
        row = {"component": name}
        for target in TARGETS:
            row[f"{target}_mean"] = float(df[target].mean())
            for year in sorted(years.unique()):
                row[f"{target}_{year}_mean"] = float(df.loc[years == year, target].mean())
        rows.append(row)

    corr = {}
    for target in TARGETS:
        mat = pd.DataFrame({name: df[target].to_numpy() for name, df in dfs.items()})
        corr[target] = mat.corr().round(5).to_dict()

    return {"component_summary": rows, "component_correlations": corr}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = load_sales()
    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}
    dfs = available_components(levels)

    files = {}
    for filename, weights in BLENDS.items():
        path = OUT / filename
        export_submission(blend(dfs, weights), path)
        files[filename] = str(path)

    # Short stable filename for Kaggle CLI.
    recommended = OUT / "m5b50300515.csv"
    shutil.copyfile(OUT / "m5blend_50_30_05_15.csv", recommended)
    files[recommended.name] = str(recommended)

    audit = {
        "levels": levels,
        "components": {name: str(path) for name, path in COMPONENTS.items()},
        "optional_components": {name: str(path) for name, path in OPTIONAL_RAW_COMPONENTS.items()},
        "available_components": sorted(dfs),
        "blends": BLENDS,
        "files": files,
        "recommended_first_submit": str(recommended),
        "leakage_policy": [
            "does not read sample_submission.csv",
            "all component files were generated from train CSVs only",
            "raw optional components are normalised using sales.csv train-derived regime_recovery levels",
        ],
        **diagnostics(dfs),
    }
    with open(OUT / "m5_style_blend_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=float)
    print(json.dumps(audit, indent=2, default=float))


if __name__ == "__main__":
    main()
