"""Driver — runs the full v3 forecasting pipeline and writes all artifacts.

Outputs (under results/v3/):
  - model_comparison.csv   (rolling-origin CV metrics × models × folds)
  - feature_importance.csv (LightGBM gain-based importances for both targets)
  - submission.csv         (548 rows, Date/Revenue/COGS, matching sample order)
  - summary.md             (chosen model, CV table, uplift, top features)

Run from repo root:
    uv run python results/v3/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the pipeline module importable no matter the cwd.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from results.v3.pipeline import (  # noqa: E402
    FOLDS,
    REGIME_START,
    TRAIN_END,
    build_daily_panel,
    feature_importance,
    fit_final_and_predict,
    load_sample_submission,
    rolling_cv,
)


OUT_DIR = HERE


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def run():
    print("=" * 72)
    print("Datathon 2026 — v3 forecasting pipeline")
    print("=" * 72)

    t0 = time.time()
    panel = build_daily_panel()
    sample = load_sample_submission()
    print(f"[load] panel={panel.shape}  sample={sample.shape}  ({time.time() - t0:.1f}s)")
    print(f"       train window: {REGIME_START.date()} → {TRAIN_END.date()}")
    print(f"       test window:  {sample['Date'].min().date()} → {sample['Date'].max().date()}")

    # -------------------- Cross-validation -------------------- #
    cv_results: dict[str, pd.DataFrame] = {}
    ens_weights: dict[str, dict[str, float]] = {}
    stashes: dict[str, dict] = {}
    for target in ("Revenue", "COGS"):
        print(f"\n[cv] rolling-origin CV for {target} …")
        t1 = time.time()
        res, stash = rolling_cv(panel, target, return_fold_preds=True)
        cv_results[target] = res.assign(target=target)
        ens_weights[target] = stash["ensemble_weights"]
        stashes[target] = stash
        print(f"     done in {time.time() - t1:.1f}s; weights={ens_weights[target]}")
        per_model = res.groupby("model")[["MAE", "RMSE", "R2", "Uplift_MAE_%"]].mean()
        print(per_model.to_string())

    all_cv = pd.concat(cv_results.values(), ignore_index=True)
    all_cv.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(f"\n[save] model_comparison.csv ← {len(all_cv)} rows")

    # -------------------- Final training & inference -------------------- #
    submission = sample[["Date"]].copy()
    importances: dict[str, pd.DataFrame] = {}
    models: dict[str, object] = {}
    feat_cols_by_target: dict[str, list[str]] = {}
    for target in ("Revenue", "COGS"):
        print(f"\n[final] fit & predict {target} on {REGIME_START.date()} → {TRAIN_END.date()}")
        t1 = time.time()
        preds, model, fcols = fit_final_and_predict(
            panel, sample, target, ens_weights[target]
        )
        print(f"     done in {time.time() - t1:.1f}s")
        submission[target] = np.round(preds, 2)
        fi = feature_importance(model, fcols).assign(target=target)
        importances[target] = fi
        models[target] = model
        feat_cols_by_target[target] = fcols
        top5 = fi.head(5)[["feature", "gain"]]
        print("     top-5 features by gain:")
        for _, r in top5.iterrows():
            print(f"       {r['feature']:<40s} gain={r['gain']:,.0f}")

    fi_all = pd.concat(importances.values(), ignore_index=True)
    fi_all.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"\n[save] feature_importance.csv ← {len(fi_all)} rows")

    # -------------------- SHAP analysis -------------------- #
    # Gain-based feature_importance is split-based; SHAP adds direction-
    # aware magnitude. We compute mean|SHAP| for both targets on a sample
    # of train-window rows, so the technical report has a second view.
    try:
        import shap  # heavy optional dep
        from results.v3.pipeline import add_features as _add_features
        shap_out = []
        for target in ("Revenue", "COGS"):
            print(f"\n[shap] SHAP values for {target} model …")
            t1 = time.time()
            feats_t, fcols_t = _add_features(panel, target)
            sample_X = (
                feats_t.loc[REGIME_START:TRAIN_END, fcols_t]
                .dropna()
                .sample(n=500, random_state=0)
            )
            expl = shap.TreeExplainer(models[target])
            sv = expl.shap_values(sample_X)
            mean_abs = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame(
                {"feature": fcols_t, "mean_abs_shap": mean_abs, "target": target}
            ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_out.append(shap_df)
            print(f"     done in {time.time() - t1:.1f}s; top-5:")
            for _, r in shap_df.head(5).iterrows():
                print(f"       {r['feature']:<40s} mean|SHAP|={r['mean_abs_shap']:.4f}")
        shap_all = pd.concat(shap_out, ignore_index=True)
        shap_all.to_csv(OUT_DIR / "shap_importance.csv", index=False)
        print("[save] shap_importance.csv")
    except Exception as e:
        print(f"[shap] skipped: {e}")

    # -------------------- Validate and save submission -------------------- #
    assert len(submission) == 548, f"submission has {len(submission)} rows, expected 548"
    assert list(submission.columns) == ["Date", "Revenue", "COGS"]
    assert submission.isnull().sum().sum() == 0
    assert (submission[["Revenue", "COGS"]] >= 0).all().all()
    # Order must match sample_submission exactly
    pd.testing.assert_series_equal(
        submission["Date"].reset_index(drop=True).astype("datetime64[ns]"),
        sample["Date"].reset_index(drop=True).astype("datetime64[ns]"),
        check_names=False,
    )
    submission_out = submission.copy()
    submission_out["Date"] = submission_out["Date"].dt.strftime("%Y-%m-%d")
    submission_out.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"[save] submission.csv ← {len(submission_out)} rows")

    # -------------------- summary.md -------------------- #
    lines: list[str] = []
    lines.append("# Summary — v3 Forecasting Pipeline\n")
    lines.append("**Team:** Data Science Team  \n**Competition:** Datathon 2026 — The Gridbreakers — Part 3 (Kaggle)\n")
    lines.append("## 1. Chosen model\n")
    lines.append("**Weighted ensemble** of three base forecasters, with weights tuned on the concatenated rolling-origin CV MAE:")
    for t in ("Revenue", "COGS"):
        w = ens_weights[t]
        lines.append(
            f"  - **{t}**: {w['lgbm']:.2f}·LightGBM + {w['sarimax']:.2f}·SARIMAX + {w['naive']:.2f}·SeasonalNaive"
        )
    lines.append(
        "\nLightGBM is the dominant component on both targets (posted the largest uplift over seasonal-naive on both folds). SARIMAX received weight 0 because its CV-MAE is substantially worse than both LightGBM and seasonal-naive; it remained in CV as an interpretable diagnostic. A small seasonal-naive component provides a regularising floor during regime-break uncertainty (Insight 1).\n"
    )

    lines.append("## 2. Rolling-origin CV (548-day validation windows)\n")
    lines.append(
        "Two folds chosen to respect the 2019 regime break (Insight 1 of `results/v2/report.md`):"
    )
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(FOLDS, start=1):
        lines.append(f"  - Fold {i}: train {tr_s.date()} → {tr_e.date()}  |  val {va_s.date()} → {va_e.date()} (548 days)")
    lines.append("")
    lines.append(
        all_cv.pivot_table(index=["target", "model"], columns="fold", values="MAE")
        .round(0).rename(columns=lambda c: f"MAE_fold{c}").to_markdown()
    )
    lines.append("\n**Fold-averaged metrics per model:**\n")
    avg = (
        all_cv.groupby(["target", "model"])[["MAE", "RMSE", "R2", "Uplift_MAE_%"]]
        .mean()
        .round({"MAE": 0, "RMSE": 0, "R2": 3, "Uplift_MAE_%": 2})
    )
    lines.append(avg.to_markdown())
    lines.append("")

    lines.append("## 3. Uplift vs seasonal-naive\n")
    naive_avg = (
        all_cv[all_cv["model"] == "seasonal_naive"]
        .groupby("target")[["MAE", "RMSE"]].mean()
    )
    for t in ("Revenue", "COGS"):
        ens_row = avg.loc[(t, "ensemble_tuned")]
        lgb_row = avg.loc[(t, "lightgbm")]
        lines.append(
            f"- **{t}**: ensemble MAE {_fmt_money(ens_row['MAE'])} vs seasonal-naive "
            f"{_fmt_money(naive_avg.loc[t, 'MAE'])} → **{ens_row['Uplift_MAE_%']:.1f}% uplift** "
            f"(LightGBM alone: {lgb_row['Uplift_MAE_%']:.1f}%)."
        )
    lines.append("")

    lines.append("## 4. Top-5 features by LightGBM gain\n")
    for t in ("Revenue", "COGS"):
        lines.append(f"**{t}:**")
        for _, r in importances[t].head(5).iterrows():
            lines.append(f"  1. `{r['feature']}` — gain {r['gain']:,.0f}, splits {int(r['split'])}")
        lines.append("")

    lines.append("## 5. Known limitations / risks\n")
    lines.append(
        "- **Regime-break risk.** 2023 may deviate further from 2019–2022 if macro or category mix shifts; our time-decay is implicit (we drop pre-2019 entirely).\n"
        "- **Recursive lag feeding.** For 2024 test dates, `Revenue_lag_364` is sourced from model predictions, which compounds errors further into the horizon.\n"
        "- **Exogenous features use climatology + lag-365/548.** If 2023 traffic/promo behaviour diverges strongly from 2022, these become stale.\n"
        "- **Stockout censoring.** Observed revenue is a lower bound of demand when stockout is high; model does not correct for this (would need Tobit/hurdle formulation).\n"
        "- **No external data.** Lunar calendar (Tết) and VN macro indicators are not included; only the `holidays` package Gregorian public holidays.\n"
        "- **Small CV folds.** Only 2 folds were used (data availability after regime cut); weight tuning may be slightly optimistic.\n"
    )

    lines.append("## 6. Reproducibility\n")
    lines.append(
        "- Source: `results/v3/pipeline.py` (features + models) and `results/v3/run_pipeline.py` (driver).\n"
        "- Runtime: ~30–40 s on a laptop CPU.\n"
        "- Deterministic: all randomness seeded (`seed=42`).\n"
    )

    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print("[save] summary.md")

    # Stash weights as JSON for downstream use
    (OUT_DIR / "ensemble_weights.json").write_text(json.dumps(ens_weights, indent=2))
    print("[save] ensemble_weights.json")

    total = time.time() - t0
    print(f"\nAll done in {total:.1f}s. Artifacts written to {OUT_DIR}")
    return submission


if __name__ == "__main__":
    run()
