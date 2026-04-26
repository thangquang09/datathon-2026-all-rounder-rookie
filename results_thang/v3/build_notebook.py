"""Generate modeling.ipynb from the pipeline scripts.

The notebook imports from pipeline.py so we keep a single source of truth.
Run: uv run python results/v3/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }


def main():
    cells = []

    cells.append(md(
        "# v3 — Daily Revenue & COGS Forecasting\n"
        "\n"
        "Datathon 2026 — *The Gridbreakers* — Part 3 (Kaggle).\n"
        "\n"
        "**Goal.** Predict daily `Revenue` and `COGS` for 2023-01-01 → 2024-07-01 (548 days).\n"
        "\n"
        "**Approach.**\n"
        "1. Seasonal-naive baseline (Rev(t) = Rev(t − 364)) — mandatory floor.\n"
        "2. SARIMAX(2,1,2) on log-revenue with weekly + yearly Fourier exog.\n"
        "3. LightGBM on log target with calendar + target lags (364, 365, 728) + exogenous climatology/lag features.\n"
        "4. Weighted ensemble tuned on CV MAE.\n"
        "\n"
        "All feature engineering, training, validation, and inference logic lives in `pipeline.py` "
        "so this notebook stays linear and reproducible. Run top-to-bottom with 'Run All'.\n"
    ))

    cells.append(md(
        "## 1. Setup\n"
        "Import the pipeline module and set up the project paths."
    ))
    cells.append(code(
        "import os\n"
        "import sys\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Ensure we run from the repo root so data/ paths resolve.\n"
        "if os.path.basename(os.getcwd()) == 'v3':\n"
        "    os.chdir('../..')\n"
        "elif os.path.basename(os.getcwd()) == 'results':\n"
        "    os.chdir('..')\n"
        "sys.path.insert(0, os.getcwd())\n"
        "print('cwd:', os.getcwd())\n"
    ))
    cells.append(code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from results.v3.pipeline import (\n"
        "    REGIME_START, TRAIN_END, FOLDS,\n"
        "    build_daily_panel, load_sample_submission,\n"
        "    add_features, rolling_cv,\n"
        "    fit_final_and_predict, feature_importance,\n"
        ")\n"
    ))

    cells.append(md(
        "## 2. Load data and build the daily panel\n"
        "We join `sales.csv` with aggregated exogenous signals (orders, web traffic, returns, inventory, promotions). "
        "Every signal is stored at daily resolution; lagging is applied later in `add_features` so the raw panel "
        "stays lossless."
    ))
    cells.append(code(
        "panel = build_daily_panel()\n"
        "sample = load_sample_submission()\n"
        "print('Panel shape :', panel.shape)\n"
        "print('Train window:', panel.index.min().date(), '->', panel.index.max().date())\n"
        "print('Test window :', sample['Date'].min().date(), '->', sample['Date'].max().date())\n"
        "panel.head(3)\n"
    ))

    cells.append(md(
        "### Regime check (Insight 1 of the EDA)\n"
        "The 2019 structural break in the revenue series drives our decision to cut training at 2019-01-01. "
        "Visual confirmation below."
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(13, 4))\n"
        "panel['Revenue'].rolling(30).mean().plot(ax=ax, lw=0.8, label='30d MA')\n"
        "panel['Revenue'].rolling(365).mean().plot(ax=ax, lw=2.0, label='365d MA', color='black')\n"
        "ax.axvline(REGIME_START, color='red', ls='--', label='regime cut (2019-01-01)')\n"
        "ax.set_title('Revenue — rolling mean; red line = start of training window used by the model')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
    ))

    cells.append(md(
        "## 3. Feature engineering\n"
        "\n"
        "The pipeline builds the following feature groups (all forecast-time safe):\n"
        "\n"
        "- **Calendar.** `dow`, `month`, `day`, `quarter`, `week_of_year`, `day_of_year`, boundary flags, VN public holidays.\n"
        "- **Fourier.** K=2 weekly (period 7) + K=4 yearly (365.25) — captures May peak / Dec trough (Insight 2).\n"
        "- **Target lags.** `{target}_lag_{364,365,728}` and `{target}_roll_{mean,std}_{W}_lag364` for W∈{7,28,91,364}.\n"
        "- **Exogenous climatology.** Day-of-year average (2019–2022) of n_orders, avg_basket, promo_share, total_discount, web signals, refund, inventory ratios. Climatology is the only exogenous value guaranteed to be valid for every test date.\n"
        "- **Exogenous lag-365 and lag-548.** To carry recent-year momentum (lag-365 is available for 2023 dates) and a strictly-safe fallback for 2024 dates (lag-548).\n"
        "\n"
        "**Leakage checklist.** We do NOT use same-day `n_orders`, `COGS`, `refund`, or any rolling statistic that sees future data. Inventory is reduced to lagged values only (≥ 365 days). See `pipeline.add_features` for the full list."
    ))
    cells.append(code(
        "feats_rev, fcols_rev = add_features(panel, 'Revenue')\n"
        "print(f'Feature matrix: {feats_rev.shape}; {len(fcols_rev)} features')\n"
        "print('Sample columns:'); print(fcols_rev[:12], '...')\n"
    ))

    cells.append(md(
        "## 4. Rolling-origin cross-validation\n"
        "\n"
        "Two folds of 548 days each, respecting the 2019 regime cut (see EDA Insight 9):\n"
        "\n"
        "| Fold | Train | Validation |\n"
        "|------|-------|------------|\n"
        "| 1 | 2019-01-01 → 2020-06-30 | 2020-07-01 → 2021-12-30 (548 d) |\n"
        "| 2 | 2019-01-01 → 2021-06-30 | 2021-07-01 → 2022-12-30 (548 d) |\n"
        "\n"
        "Every model is compared against seasonal-naive; the `ensemble_tuned` row uses CV-optimal weights."
    ))
    cells.append(code(
        "cv_rev, stash_rev = rolling_cv(panel, 'Revenue', return_fold_preds=True)\n"
        "cv_cogs, stash_cogs = rolling_cv(panel, 'COGS', return_fold_preds=True)\n"
        "cv_all = pd.concat([cv_rev.assign(target='Revenue'), cv_cogs.assign(target='COGS')], ignore_index=True)\n"
        "print('Revenue ensemble weights:', stash_rev['ensemble_weights'])\n"
        "print('COGS ensemble weights:   ', stash_cogs['ensemble_weights'])\n"
        "display(cv_all.groupby(['target','model'])[['MAE','RMSE','R2','Uplift_MAE_%']].mean().round(2))\n"
    ))
    cells.append(code(
        "for t in ('Revenue', 'COGS'):\n"
        "    df = cv_all[cv_all.target==t].pivot_table(index='model', columns='fold', values='MAE').round(0)\n"
        "    print(f'=== {t} — MAE per fold ==='); print(df); print()\n"
    ))

    cells.append(md(
        "## 5. Final training on 2019-01-01 → 2022-12-31 and 548-day inference\n"
        "\n"
        "We retrain each base model on the full post-regime training window, then combine with the CV-tuned weights. "
        "LightGBM uses **recursive** substitution for `{target}_lag_364` on 2024 test dates (the lag source lies in "
        "2023, which is itself test data)."
    ))
    cells.append(code(
        "submission = sample[['Date']].copy()\n"
        "importances = {}\n"
        "models = {}\n"
        "for target in ('Revenue', 'COGS'):\n"
        "    w = stash_rev['ensemble_weights'] if target == 'Revenue' else stash_cogs['ensemble_weights']\n"
        "    preds, model, fcols = fit_final_and_predict(panel, sample, target, w)\n"
        "    submission[target] = np.round(preds, 2)\n"
        "    importances[target] = feature_importance(model, fcols)\n"
        "    models[target] = model\n"
        "    print(f'{target}: min={preds.min():,.0f}  max={preds.max():,.0f}  mean={preds.mean():,.0f}')\n"
        "submission.head()\n"
    ))

    cells.append(md(
        "### Quality gates — verify every requirement before writing submission.csv"
    ))
    cells.append(code(
        "assert len(submission) == 548\n"
        "assert list(submission.columns) == ['Date', 'Revenue', 'COGS']\n"
        "assert submission.isnull().sum().sum() == 0\n"
        "assert (submission[['Revenue','COGS']] >= 0).all().all()\n"
        "assert submission['Date'].reset_index(drop=True).equals(sample['Date'].reset_index(drop=True))\n"
        "print('All quality gates passed.')\n"
    ))

    cells.append(md(
        "## 6. Feature importance\n"
        "\n"
        "LightGBM gain-based importance (how much each feature reduces loss across splits) is a first-pass view. "
        "We complement it with mean |SHAP| for the top features to verify magnitude and direction."
    ))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "for ax, t in zip(axes, ('Revenue', 'COGS')):\n"
        "    top = importances[t].head(15).iloc[::-1]\n"
        "    ax.barh(top['feature'], top['gain'])\n"
        "    ax.set_title(f'{t} — top 15 LightGBM features (gain)')\n"
        "plt.tight_layout(); plt.show()\n"
    ))
    cells.append(code(
        "try:\n"
        "    import shap\n"
        "    for t in ('Revenue',):\n"
        "        feats_t, fcols_t = add_features(panel, t)\n"
        "        X_sample = feats_t.loc[REGIME_START:TRAIN_END, fcols_t].dropna().sample(500, random_state=0)\n"
        "        expl = shap.TreeExplainer(models[t])\n"
        "        sv = expl.shap_values(X_sample)\n"
        "        shap.summary_plot(sv, X_sample, max_display=15, show=False)\n"
        "        plt.title(f'{t} — SHAP summary (500 train rows)')\n"
        "        plt.tight_layout(); plt.show()\n"
        "except Exception as e:\n"
        "    print('SHAP plot skipped:', e)\n"
    ))

    cells.append(md(
        "## 7. Save submission\n"
        "Write `submission.csv` with Date in YYYY-MM-DD format, identical row order to `sample_submission.csv`."
    ))
    cells.append(code(
        "out = submission.copy()\n"
        "out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')\n"
        "out.to_csv('results/v3/submission.csv', index=False)\n"
        "print('Saved', len(out), 'rows to results/v3/submission.csv')\n"
        "out.head()\n"
    ))

    cells.append(md(
        "## 8. Sanity: predicted vs. historical seasonality\n"
        "Overlay the 548 predicted days against the last two complete historical years (2021–2022) to check that "
        "the annual shape looks right (May peak, Dec trough) and levels are in the 2022 range."
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(14, 4))\n"
        "hist = panel.loc['2021-01-01':'2022-12-31', 'Revenue']\n"
        "ax.plot(hist.index, hist.values, lw=0.6, label='Revenue 2021–2022 (observed)')\n"
        "ax.plot(submission['Date'], submission['Revenue'], lw=0.8, color='crimson', label='Predicted 2023-01 → 2024-07')\n"
        "ax.set_title('Observed vs predicted daily revenue')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = HERE / "modeling.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
