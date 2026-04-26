"""Compute SHAP values for the trained LGBM models and produce diagnostic plots.

Run after `src.lgbm_model` has written the model files under outputs/models/.
Produces:
- outputs/shap_<target>_summary.png
- outputs/shap_<target>_importance.png
- outputs/shap_<target>_dependence_<feat>.png
- outputs/shap_<target>_top_features.json
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.lgbm_model import (
    MODELS,
    build_training_frame,
    feature_columns,
    time_split,
)


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def run_shap(target: str, top_k_dep: int = 5, sample_rows: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    df = build_training_frame(target)
    feats = feature_columns(df)
    train, val = time_split(df, target)

    model_file = MODELS / f"lgbm_{target.lower()}.txt"
    if not model_file.exists():
        raise FileNotFoundError(f"Train the model first: missing {model_file}")
    booster = lgb.Booster(model_file=str(model_file))

    background = train.sample(n=min(1000, len(train)), random_state=seed)[feats]
    sample = val.sample(n=min(sample_rows, len(val)), random_state=seed)[feats]

    explainer = shap.TreeExplainer(booster, background)
    shap_values = explainer.shap_values(sample, check_additivity=False)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_feats = [(feats[i], float(mean_abs[i])) for i in order[:30]]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    plt.title(f"SHAP summary — {target}")
    plt.tight_layout()
    plt.savefig(OUTPUTS / f"shap_{target.lower()}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, plot_type="bar", max_display=25, show=False)
    plt.title(f"SHAP feature importance — {target}")
    plt.tight_layout()
    plt.savefig(OUTPUTS / f"shap_{target.lower()}_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    top_feats_dep = [f for f, _ in top_feats[:top_k_dep]]
    for f in top_feats_dep:
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(f, shap_values, sample, show=False)
            plt.title(f"SHAP dependence — {f} ({target})")
            plt.tight_layout()
            out = OUTPUTS / f"shap_{target.lower()}_dependence_{f.replace('/', '_')}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"skip dependence for {f}: {e}")

    out = {
        "target": target,
        "top_features_by_mean_abs_shap": [
            {"feature": f, "mean_abs_shap": v} for f, v in top_feats
        ],
        "num_background": int(len(background)),
        "num_samples": int(len(sample)),
    }
    with open(OUTPUTS / f"shap_{target.lower()}_top_features.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    for t in ["Revenue", "COGS"]:
        res = run_shap(t)
        print(f"\n== {t} top features ==")
        for item in res["top_features_by_mean_abs_shap"][:15]:
            print(f"  {item['feature']:40s}  mean|SHAP| = {item['mean_abs_shap']:,.0f}")
