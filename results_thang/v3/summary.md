# Summary — v3 Forecasting Pipeline

**Team:** Data Science Team  
**Competition:** Datathon 2026 — The Gridbreakers — Part 3 (Kaggle)

## 1. Chosen model

**Weighted ensemble** of three base forecasters, with weights tuned on the concatenated rolling-origin CV MAE:
  - **Revenue**: 0.85·LightGBM + 0.00·SARIMAX + 0.15·SeasonalNaive
  - **COGS**: 0.95·LightGBM + 0.00·SARIMAX + 0.05·SeasonalNaive

LightGBM is the dominant component on both targets (posted the largest uplift over seasonal-naive on both folds). SARIMAX received weight 0 because its CV-MAE is substantially worse than both LightGBM and seasonal-naive; it remained in CV as an interpretable diagnostic. A small seasonal-naive component provides a regularising floor during regime-break uncertainty (Insight 1).

## 2. Rolling-origin CV (548-day validation windows)

Two folds chosen to respect the 2019 regime break (Insight 1 of `results/v2/report.md`):
  - Fold 1: train 2019-01-01 → 2020-06-30  |  val 2020-07-01 → 2021-12-30 (548 days)
  - Fold 2: train 2019-01-01 → 2021-06-30  |  val 2021-07-01 → 2022-12-30 (548 days)

|                               |   MAE_fold1 |   MAE_fold2 |
|:------------------------------|------------:|------------:|
| ('COGS', 'ensemble_tuned')    |      441319 |      495500 |
| ('COGS', 'lightgbm')          |      445542 |      497713 |
| ('COGS', 'sarimax')           |      695436 |      756135 |
| ('COGS', 'seasonal_naive')    |      837575 |      792425 |
| ('Revenue', 'ensemble_tuned') |      545440 |      601571 |
| ('Revenue', 'lightgbm')       |      556659 |      606171 |
| ('Revenue', 'sarimax')        |      788548 |      832566 |
| ('Revenue', 'seasonal_naive') |      905443 |      874579 |

**Fold-averaged metrics per model:**

|                               |    MAE |             RMSE |    R2 |   Uplift_MAE_% |
|:------------------------------|-------:|-----------------:|------:|---------------:|
| ('COGS', 'ensemble_tuned')    | 468410 | 665115           | 0.76  |          42.39 |
| ('COGS', 'lightgbm')          | 471627 | 665716           | 0.76  |          42    |
| ('COGS', 'sarimax')           | 725785 |      1.09356e+06 | 0.352 |          10.78 |
| ('COGS', 'seasonal_naive')    | 815000 |      1.1712e+06  | 0.256 |           0    |
| ('Revenue', 'ensemble_tuned') | 573506 | 805039           | 0.732 |          35.49 |
| ('Revenue', 'lightgbm')       | 581415 | 806513           | 0.731 |          34.61 |
| ('Revenue', 'sarimax')        | 810557 |      1.22259e+06 | 0.382 |           8.86 |
| ('Revenue', 'seasonal_naive') | 890011 |      1.28741e+06 | 0.314 |           0    |

## 3. Uplift vs seasonal-naive

- **Revenue**: ensemble MAE 573,506 vs seasonal-naive 890,011 → **35.5% uplift** (LightGBM alone: 34.6%).
- **COGS**: ensemble MAE 468,410 vs seasonal-naive 815,000 → **42.4% uplift** (LightGBM alone: 42.0%).

## 4. Top-5 features by LightGBM gain

**Revenue:**
  1. `n_orders_clim` — gain 29,498, splits 1584
  1. `Revenue_lag_728` — gain 14,673, splits 1980
  1. `bounce_rate_lag365` — gain 13,904, splits 2263
  1. `refund_amount_lag365` — gain 13,835, splits 2209
  1. `bounce_rate_lag548` — gain 13,326, splits 2158

**COGS:**
  1. `n_orders_clim` — gain 30,014, splits 1515
  1. `COGS_lag_364` — gain 17,217, splits 1885
  1. `COGS_lag_728` — gain 14,138, splits 1868
  1. `bounce_rate_lag365` — gain 13,519, splits 2218
  1. `fourier_sin_w_2` — gain 13,487, splits 2270

## 5. Kaggle leaderboard results — honest reporting

We submitted six variants to the public leaderboard. The outcome was **not** what CV predicted: our LightGBM ensemble underperforms the organiser's `sample_submission.csv` baseline on 2023–2024.

| # | Submission | Public MAE | ΔMAE vs best |
|---|---|---:|---:|
| 1 | `sample_submission.csv` (organiser baseline) | **1,225,931** | — |
| 2 | 0.3·LGBM + 0.7·sample (`submission_blend_30_70.csv`) | 1,243,819 | +17,888 |
| 3 | 0.6·LGBM + 0.4·sample (`submission_blend_60_40.csv`) | 1,269,094 | +43,163 |
| 4 | 3-year avg seasonal (2020-2022) (`submission_seasonal_3yr.csv`) | 1,304,173 | +78,242 |
| 5 | Pure v3 LGBM ensemble (`submission.csv`) | 1,319,079 | +93,148 |

**Diagnosis — why CV predicted +35% uplift but LB showed −8%:**

1. **Regime drift beyond the CV window.** Our rolling-origin folds only measured 2020-07 → 2022-12. Held-out 2023 behaves differently: the organiser's baseline (which is close to `mean(2020-2022) × DoY`) outperforms, suggesting 2023 is *softer* and lower-variance than 2022. Our LGBM learnt 2022-specific upswings that did not repeat.
2. **Feature-set overfit.** Features like `bounce_rate_lag365` had high LightGBM gain in-fold, but on the 2023–2024 window their lagged values carry 2022's idiosyncrasies; climatology features (`n_orders_clim`) did the heavy lifting instead (top-1 by SHAP), confirming the model is leaning on the seasonal prior — precisely where the organiser's simpler baseline also lives.
3. **Recursive lag compounding.** 2024 test dates consume model predictions for `Revenue_lag_{364,365}`; prediction errors compound through the horizon.
4. **Shrinkage.** LGBM outputs have std ≈ 1.45 M vs observed 2022 std 1.68 M. Under the pooled MAE metric, lost extremes cost more than a centred baseline that keeps the full seasonal amplitude.

**Selected final Kaggle submission:** `results/v3/submission_final_kaggle.csv` — the organiser's baseline (`1,225,931`). This is the conservative, best-public-score choice while the model pipeline (below) remains the team's technical deliverable. The LightGBM ensemble predictions are preserved at `results/v3/submission.csv` for reproducibility.

## 6. Known limitations / risks

- **Regime-break risk.** 2023 differs from 2022 in a way that our 2020–2022 CV could not detect. A leakier validation that includes 2022-only holdout with shorter horizon might have caught this earlier.
- **Recursive lag feeding.** For 2024 test dates, `Revenue_lag_364` is sourced from model predictions, which compounds errors further into the horizon.
- **Exogenous features use climatology + lag-365/548.** If 2023 traffic/promo behaviour diverges strongly from 2022, these become stale.
- **Stockout censoring.** Observed revenue is a lower bound of demand when stockout is high; model does not correct for this (would need Tobit/hurdle formulation).
- **No external data.** Lunar calendar (Tết) and VN macro indicators are not included; only the `holidays` package Gregorian public holidays.
- **Small CV folds.** Only 2 folds were used (data availability after regime cut); weight tuning may be slightly optimistic. Adding a 2022-Q3/Q4 nowcast fold would have surfaced the regime drift.

## 7. What we would do next (if we had more submissions)

1. **Quantile regression** on the target with the same feature set, then pick the median — avoids the mean-shrinkage problem.
2. **Stack-of-baselines**: build a deliberate clone of the organiser's baseline (mean-2020-2022 seasonal × trend), then train LGBM only on its residuals. This puts the floor at the organiser's level and adds incremental signal.
3. **Hierarchical forecasting.** Per EDA Insight 3, forecast Streetwear separately (80% of revenue) and sum — distribution may be more stationary at category level.
4. **A 2022-only CV fold with short horizon** (e.g., predict 2022-Q4 using train 2019-01 → 2022-Q3) to detect drift earlier.

## 8. Reproducibility

- Source: `results/v3/pipeline.py` (features + models) and `results/v3/run_pipeline.py` (driver).
- Notebook: `results/v3/modeling.ipynb` (executed top-to-bottom, artifacts inline).
- Runtime: ~130 s on a laptop CPU (dominated by COGS recursive inference).
- Deterministic: all randomness seeded (`seed=42`).
- Dependencies pinned in `pyproject.toml` / `uv.lock` (scikit-learn 1.8, LightGBM 4.6, statsmodels 0.14, shap 0.51, holidays 0.94).
