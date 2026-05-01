"""Build legacy-model diversity blends with train-only regime recovery levels.

The repo already contains several independent LightGBM model families:
- v1: early rich exogenous model
- v2: leakage-safe log LightGBM bag
- v3: Tweedie/diverse lag set
- v4: broad feature-engineering overhaul

This script regenerates their raw forecasts, normalises each to the
train-derived `regime_recovery` yearly means, and exports weighted blends.
It does not read sample_submission.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.forecast_pipeline import (  # noqa: E402
    FORECAST_END,
    FORECAST_START,
    TARGETS,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)


OUT = ROOT / "artifacts"
SEEDS = (42, 123, 7, 2024, 31)


def build_v1_raw(dates: pd.DatetimeIndex) -> pd.DataFrame:
    from legacy_components import final_model as fm

    out = pd.DataFrame({"Date": dates})
    for target in TARGETS:
        df = fm.build_frame(target)
        feats = fm.feature_cols(df, target)
        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < "2022-01-01"]
        val = hist[hist["Date"] >= "2022-01-01"]

        dtrain = lgb.Dataset(train[feats], label=train[target])
        dval = lgb.Dataset(val[feats], label=val[target], reference=dtrain)
        model = lgb.train(
            fm.lgb_params(42),
            dtrain,
            num_boost_round=6000,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)],
        )
        dfull = lgb.Dataset(hist[feats], label=hist[target])
        full_model = lgb.train(
            fm.lgb_params(42),
            dfull,
            num_boost_round=model.best_iteration or 2000,
            valid_sets=[dfull],
            valid_names=["full"],
            callbacks=[lgb.log_evaluation(0)],
        )
        out[target] = fm.recursive_forecast(df, target, full_model, feats)
    return out


def build_v2_raw(dates: pd.DatetimeIndex) -> pd.DataFrame:
    from legacy_components.final_model_v2 import fit_and_forecast

    out = pd.DataFrame({"Date": dates})
    for target in TARGETS:
        raw, _, _, _ = fit_and_forecast(target, seeds=SEEDS)
        out[target] = raw
    return out


def build_v3_raw(dates: pd.DatetimeIndex) -> pd.DataFrame:
    from legacy_components.final_model_v3 import (
        build_frame_v3,
        feature_cols,
        train_seed,
        train_full,
        recursive_forecast,
    )

    out = pd.DataFrame({"Date": dates})
    for target in TARGETS:
        df = build_frame_v3(target)
        feats = feature_cols(df, target)
        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < "2022-01-01"]
        val = hist[hist["Date"].dt.year == 2022]

        models = []
        for seed in SEEDS:
            es = train_seed(train, val, feats, target, seed)
            rounds = int((es.best_iteration or 1500) * 1.1)
            models.append(train_full(hist, feats, target, seed, rounds))
        out[target] = recursive_forecast(df, target, models, feats)
    return out


def load_or_build_v4_raw(dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = OUT / "submission_v4_raw.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])

    from legacy_components.final_model_v4 import fit_and_forecast

    out = pd.DataFrame({"Date": dates})
    for target in TARGETS:
        raw, _, _, _ = fit_and_forecast(target, seeds=SEEDS)
        out[target] = raw
    export_submission(out, path)
    return out


def blend(dfs: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    date = next(iter(dfs.values()))["Date"]
    out = pd.DataFrame({"Date": date})
    for target in TARGETS:
        out[target] = sum(weights[k] * dfs[k][target].to_numpy() for k in weights)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    sales = load_sales()
    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}

    raw_builders = {
        "v1": build_v1_raw,
        "v2": build_v2_raw,
        "v3": build_v3_raw,
        "v4": load_or_build_v4_raw,
    }

    raw = {}
    normalised = {}
    for key, builder in raw_builders.items():
        raw_path = OUT / f"legacy_{key}_raw.csv"
        norm_path = OUT / f"legacy_{key}_regime_recovery.csv"
        if raw_path.exists() and key != "v1":
            df = pd.read_csv(raw_path, parse_dates=["Date"])
        else:
            print(f"Building {key} raw forecast ...")
            df = builder(dates)
            export_submission(df, raw_path)
        raw[key] = df
        normalised[key] = normalise_yearly(df, levels)
        export_submission(normalised[key], norm_path)

    blends = {
        "legacy_blend_50_30_05_15.csv": {"v1": 0.50, "v2": 0.30, "v3": 0.05, "v4": 0.15},
        "legacy_blend_50_35_00_15.csv": {"v1": 0.50, "v2": 0.35, "v3": 0.00, "v4": 0.15},
        "legacy_blend_45_35_05_15.csv": {"v1": 0.45, "v2": 0.35, "v3": 0.05, "v4": 0.15},
        "legacy_blend_55_30_00_15.csv": {"v1": 0.55, "v2": 0.30, "v3": 0.00, "v4": 0.15},
    }
    files = {}
    for filename, weights in blends.items():
        out = blend(normalised, weights)
        path = OUT / filename
        export_submission(out, path)
        files[filename] = str(path)

    summary = {
        "levels": levels,
        "raw_files": {k: str(OUT / f"legacy_{k}_raw.csv") for k in raw_builders},
        "blend_files": files,
        "recommended_first_submit": files["legacy_blend_50_30_05_15.csv"],
        "leakage_policy": [
            "all component models are trained from train CSVs only",
            "legacy v1 is rebuilt with target-proxy same-day features dropped",
            "no sample_submission read in this script",
            "yearly normalisation uses regime_recovery levels derived from sales.csv",
        ],
    }
    with open(OUT / "legacy_blend_regime.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
