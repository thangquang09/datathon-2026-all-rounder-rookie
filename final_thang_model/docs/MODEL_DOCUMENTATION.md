# Pipeline Documentation — `final_thang_model`

> **Datathon 2026 — Phan 3: Du bao Doanh thu (Sales Forecasting)**
>
> Forecast daily `Revenue` and `COGS` for the period **2023-01-01 to 2024-07-01** (548 days).

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Data Sources](#2-data-sources)
3. [Feature Engineering](#3-feature-engineering)
   - 3.1 [Calendar & Seasonality Features](#31-calendar--seasonality-features)
   - 3.2 [Vietnamese Holiday & E-commerce Event Features](#32-vietnamese-holiday--e-commerce-event-features)
   - 3.3 [Target Lag & Rolling Features](#33-target-lag--rolling-features)
   - 3.4 [Exogenous Features from 13 Operational CSVs](#34-exogenous-features-from-13-operational-csvs)
   - 3.5 [Level-Leak Column Handling](#35-level-leak-column-handling)
   - 3.6 [Level Calibration Features](#36-level-calibration-features)
4. [Leakage Prevention Policy](#4-leakage-prevention-policy)
5. [Model Architecture](#5-model-architecture)
   - 5.1 [Pipeline A: Recursive LightGBM Ensemble](#51-pipeline-a-recursive-lightgbm-ensemble)
   - 5.2 [Pipeline B: Direct Multi-Step Model](#52-pipeline-b-direct-multi-step-model)
   - 5.3 [Pipeline C: Legacy Model Versions (v1–v4)](#53-pipeline-c-legacy-model-versions-v1v4)
   - 5.4 [Final Blend: M5-Style 80/20](#54-final-blend-m5-style-8020)
6. [Training & Validation](#6-training--validation)
   - 6.1 [Time-Aware Cross-Validation](#61-time-aware-cross-validation)
   - 6.2 [Walk-Forward CV Details](#62-walk-forward-cv-details)
7. [Yearly Level Calibration](#7-yearly-level-calibration)
8. [Explainability](#8-explainability)
9. [Results Comparison](#9-results-comparison)
10. [File Structure](#10-file-structure)
11. [How to Reproduce](#11-how-to-reproduce)

---

## 1. Overview & Architecture

The final submission is an **80/20 weighted blend** of two independent forecast systems:

```
Final Submission (submission.csv)
├── 80% — M5-Style Multi-Version Blend
│   ├── 50% Legacy v1-no-proxy (regime_recovery calibrated)
│   ├── 30% Legacy v2 (regime_recovery calibrated)
│   ├──  5% Legacy v3 (regime_recovery calibrated)
│   └── 15% Legacy v4 (regime_recovery calibrated)
│
└── 20% — Direct LGBM+Ridge Ensemble
    ├── LightGBM (direct multi-step, regime_recovery calibrated)
    ├── RidgeCV (direct multi-step)
    └── DoY Prior (seasonal baseline)
```

**Key design principles:**
- **Leakage-safe**: No Revenue/COGS from test period or sample_submission used
- **Train-only data**: All features derived from the 13 provided train CSVs
- **Reproducible**: Fixed seeds, deterministic LightGBM, auditable artifact trail
- **Explainable**: SHAP values, feature importance, feature-group business explanations

---

## 2. Data Sources

| File | Period | Usage |
|---|---|---|
| `sales.csv` | 2012-07-04 to 2022-12-31 | Target (Revenue, COGS) + yearly level calibration |
| `sample_submission.csv` | 2023-01-01 to 2024-07-01 | **Date column only** (row order verification) |
| `orders.csv` | 2012-2022 | Daily order counts, status shares, device/source mix |
| `order_items.csv` | 2012-2022 | Product mix, quantities, discounts, promo usage |
| `products.csv` | — | Category, segment, size, price, COGS per product |
| `payments.csv` | 2012-2022 | Payment method mix, installment structure |
| `shipments.csv` | 2012-2022 | Shipping leadtime, fee, shipment volume |
| `returns.csv` | 2012-2022 | Return counts, quantities, refund amounts |
| `reviews.csv` | 2012-2022 | Review counts, rating averages, bad review rate |
| `web_traffic.csv` | 2012-2022 | Sessions, visitors, page views, bounce rate |
| `promotions.csv` | 2012-2022 | Active promo counts, discount depth, promo timing |
| `inventory.csv` | 2012-2022 (monthly) | Stockout, overstock, fill rate, sell-through |
| `customers.csv` | 2012-2022 | Signup dates, acquisition channel, demographics |
| `geography.csv` | — | Region/district mapping for order zip codes |
| `vietnam_calendar_events_deterministic_2012_2024.csv` | 2012-2024 | Machine-readable calendar events generated from Gregorian rules |

**Crucially**: `sales_test.csv` is never read. `sample_submission.csv` is read with `usecols=['Date']` only.

---

## 3. Feature Engineering

### 3.1 Calendar & Seasonality Features

Built from the `Date` index — these are **fully deterministic** and available for any future date.

| Feature | Description | Type |
|---|---|---|
| `year`, `month`, `day`, `dow`, `doy`, `week`, `quarter` | Standard calendar decompositions | Integer |
| `is_weekend` | Saturday/Sunday flag (0/1) | Binary |
| `is_month_start`, `is_month_end` | Month boundary flags | Binary |
| `is_payday_window` | Day 25–5 of next month (Vietnamese salary cycle) | Binary |
| `is_midmonth_window` | Day 13–17 (mid-month bonus cycle) | Binary |
| `days_since_start` | Linear time counter from first date | Integer |
| `post_regime` | Flag for post-2019 structural break | Binary |
| `sin_week_k`, `cos_week_k` (k=1,2,3) | Fourier harmonics for weekly seasonality | Float |
| `sin_year_k`, `cos_year_k` (k=1..6 or 8) | Fourier harmonics for annual seasonality | Float |

**Why**: Fashion e-commerce has strong weekly patterns (weekday vs weekend) and annual patterns (seasonal collections, holidays). Fourier harmonics capture smooth cyclic behavior without creating 365 one-hot columns.

### 3.2 Vietnamese Holiday & E-commerce Event Features

Built from `src/calendar_vn.py` using deterministic rules from the Gregorian
`Date` field. Fixed-date events use month/day checks, Black Friday uses the
last-Friday-of-November rule, and lunar events use Vietnamese solar-to-lunar
conversion with UTC+7.

**Lunar festivals** (dates computed from Gregorian dates via solar-to-lunar conversion):
- **Tet Nguyen Dan** (Lunar New Year): `days_to_tet`, `days_since_tet`, `is_tet_window`, `is_tet_eve_7d`, `is_tet_after_14d`
- **Hung Kings Commemoration Day** (Lunar Mar 10): `days_to_hung_kings`, `is_hung_kings_window`
- **Mid-Autumn Festival** (Lunar Aug 15): `days_to_mid_autumn`, `is_mid_autumn_eve_7d`

**Fixed-date national holidays**:
- Reunification Day (Apr 30), Labour Day (May 1), Independence Day (Sep 2), New Year (Jan 1)

**E-commerce shopping festivals**:
- 11/11 (Singles Day): `is_1111`, `days_to_1111`, `is_1111_week`
- 12/12: `is_1212`, `days_to_1212`, `is_1212_week`
- Black Friday (last Friday of Nov): `days_to_black_friday`, `is_black_friday_week`
- Cyber Monday: `is_cyber_monday_window`

**Other gifting/observance days**:
- Valentine (Feb 14), Women's Day (Mar 8, Oct 20), Teachers' Day (Nov 20), Christmas (Dec 24-25)

**Why**: Vietnamese e-commerce has pronounced demand spikes around Tet (the biggest annual slowdown + pre-Tet gifting surge), 11/11/12/12 festivals, and Black Friday. Distance-based features (`days_to_*`, `days_since_*`) let the model learn pre-event ramp-up and post-event dip patterns.

### 3.3 Target Lag & Rolling Features

These capture the **autoregressive structure** of the time series.

**Lag features** (values from exactly N days ago):
| Lag | Rationale |
|---|---|
| 7, 14 | Recent short-term level |
| 28, 56 | Monthly-ish memory |
| 91 | Quarterly pattern |
| 182 | Half-year |
| 364, 365, 371 | Year-over-year exact + ±1 week anchor |
| 548 | Full forecast horizon anchor |
| 728, 730 | Two-year memory |

**Rolling features** (statistics over a trailing window, shifted by 1 day to avoid same-day leakage):
| Window | Features | Rationale |
|---|---|---|
| 7, 14, 28, 56, 91, 182, 365 | `roll_mean`, `roll_std` | Recent trend and volatility |
| 7, 28, 56, 91, 182, 365 | `season_mean_lag364` | Same-season anchor from one year ago |

**Derived features**:
- `{target}_lag365_div_lag730`: Year-over-year growth ratio
- `{target}_anchor_latest`: Most recent known value at cutoff
- `{target}_anchor_yoy_ratio365`: YoY change at cutoff
- `{target}_recent_vs_hist`: Recent period mean vs full history ratio

**How these are computed for the forecast horizon**:
- **Recursive pipeline**: Lags are filled step-by-step — each day's prediction becomes the lag input for subsequent days
- **Direct pipeline**: All lags are computed from **known history only** at each cutoff date, with no recursive feedback

### 3.4 Exogenous Features from 13 Operational CSVs

All daily aggregates are computed from the train-period CSVs. For the forecast horizon, values are replaced by **day-of-year mean climatology** from the training period.

#### Orders (`orders.csv`)
| Feature | Description |
|---|---|
| `n_orders` | Daily order count |
| `n_customers` | Unique customers per day |
| `n_zips` | Unique delivery locations per day |
| `delivered_share`, `cancelled_share`, `returned_status_share` | Order status proportions |
| `device_mobile_share`, `device_desktop_share`, `device_tablet_share` | Device type distribution |
| `source_organic_search_share`, `source_paid_search_share`, etc. | Marketing channel distribution |

#### Order Items (`order_items.csv` + `products.csv`)
| Feature | Description |
|---|---|
| `item_qty` | Total items sold per day |
| `item_lines` | Number of line items |
| `item_unique_products` | Product diversity |
| `item_avg_unit_price` | Average unit price |
| `item_discount` | Total discount amount |
| `item_promo_share` | Share of items with promotions |
| `item_size_lxl_share` | Size mix (L/XL proportion) |
| `category_streetwear_qty_share`, etc. | Category mix proportions |
| `segment_everyday_qty_share`, etc. | Segment mix proportions |
| `avg_basket_qty` | Items per order |
| `avg_basket_value` | Revenue per order |
| `discount_rate` | Discount as % of gross |

#### Payments (`payments.csv`)
| Feature | Description |
|---|---|
| `pay_installments_mean` | Average installment count |
| `pay_install_gt3_share` | Share of orders with >3 installments |
| `pay_credit_card_share`, `pay_paypal_share`, etc. | Payment method distribution |

#### Web Traffic (`web_traffic.csv`)
| Feature | Description |
|---|---|
| `web_sessions` | Total daily sessions |
| `web_unique_visitors` | Unique visitors |
| `web_page_views` | Page views |
| `web_bounce_rate` | Average bounce rate |
| `web_avg_session` | Average session duration (seconds) |
| `web_pv_per_session` | Page views per session |

#### Returns (`returns.csv`)
| Feature | Description |
|---|---|
| `returns_count` | Daily return count |
| `returns_qty` | Total returned quantity |
| `returns_refund` | Total refund amount |

#### Reviews (`reviews.csv`)
| Feature | Description |
|---|---|
| `reviews_count` | Daily review count |
| `reviews_rating_mean` | Average rating |
| `reviews_bad_rate` | Share of ratings ≤ 2 |

#### Shipments (`shipments.csv`)
| Feature | Description |
|---|---|
| `ship_count` | Daily shipments |
| `ship_fee_mean` | Average shipping fee |
| `ship_leadtime_mean` | Average delivery leadtime (days) |

#### Inventory (`inventory.csv`)
| Feature | Description |
|---|---|
| `inv_stockout_rate` | Average stockout flag across products |
| `inv_overstock_rate` | Average overstock flag |
| `inv_reorder_rate` | Average reorder flag |
| `inv_fill_rate` | Average fill rate |
| `inv_sell_through` | Average sell-through rate |
| `inv_days_of_supply` | Average days of supply |

#### Promotions (`promotions.csv`)
| Feature | Description |
|---|---|
| `promo_active_count` | Number of active promos on a given day |
| `promo_max_discount` | Maximum discount among active promos |
| `promo_mean_discount` | Average discount among active promos |
| `promo_active` | Binary: any promo active? |
| `days_since_last_promo_start` | Recency of last promo |
| `days_to_next_promo_start` | Proximity to next promo |

#### Customer Lifecycle (`customers.csv` + `orders.csv`)
| Feature | Description |
|---|---|
| `first_time_orders` | New customer orders |
| `repeat_orders` | Returning customer orders |
| `first_time_buyer_rate` | New customer share |
| `customer_age_mean` | Average customer tenure |
| `active_customers_28d`, `active_customers_90d`, `active_customers_365d` | Rolling unique active customer count |

#### Geography (`geography.csv` + `orders.csv`)
| Feature | Description |
|---|---|
| `orders_unique_cities` | Unique delivery cities per day |
| `orders_region_east_share`, `orders_region_central_share`, etc. | Regional distribution |

### 3.5 Level-Leak Column Handling

Some daily aggregates are **semantically equivalent to the target**:

| Column | Problem |
|---|---|
| `items_cogs_total_value` | = COGS exactly (sum of quantity × product COGS) |
| `items_gross_value` | ≈ Revenue (sum of quantity × unit_price) |
| `pay_total_value` | ≈ Revenue × 1.06 |
| `pay_mean_value` | Strongly correlated with Revenue |
| `items_discount_total` | Scales linearly with Revenue |
| `orders_count` | Near-perfect correlation (|r| > 0.94) with Revenue |

**Handling**:

- Legacy `v1` drops the same-day target-proxy columns and their short-lag/rolling
  derivatives from `feature_cols()`.
- v2/v3 drop the direct proxy columns and replace volatile operational signals
  with train-period day-of-year means.
- v4 replaces the full exogenous frame with train-period day-of-year means using
  `replace_with_doy_mean()`, preserving seasonal shape without exact daily
  values.

### 3.6 Level Calibration Features

These features help the model understand the **overall demand level** relative to the historical regime:

| Feature | Description |
|---|---|
| `{target}_doy_mean_train` | Historical day-of-year mean (train-cutoff only) |
| `{target}_doy_median_train` | Historical day-of-year median |
| `{target}_month_mean_train` | Historical monthly mean |
| `{target}_dow_mean_train` | Historical day-of-week mean |
| `{target}_month_dow_mean_train` | Historical month×weekday interaction mean |
| `{target}_annual_mean_cutoff` | Last known annual mean at cutoff |
| `{target}_pre_break_mean_cutoff` | Mean of pre-2019 (pre-break) period |
| `{target}_post_break_mean_cutoff` | Mean of post-2019 (post-break) period |
| `{target}_post_to_pre_ratio` | Ratio of post-break to pre-break level |
| `post_regime` | Binary flag: is this date after the 2019 structural break? |

---

## 4. Leakage Prevention Policy

The following explicit safeguards are implemented:

1. **`sample_submission.csv` is read with `usecols=['Date']`** — Revenue/COGS columns are never loaded into memory
2. **Future exogenous rows are blanked**: All exogenous columns for dates after `TRAIN_END` (2022-12-31) are set to `NaN` before feature construction
3. **Day-of-year exogenous climatology** is computed only from data up to each training cutoff, then applied to forecast dates
4. **Target lags for the forecast horizon** are filled recursively from prior predictions (not from actual values)
5. **Direct model features** respect cutoff boundaries — `_series_get()` returns NaN for dates after the cutoff
6. **Yearly level calibration** uses only `sales.csv` historical aggregates (regime_recovery levels)
7. **Legacy v1 target-proxy guard** drops same-day operational proxies such as `items_gross_value`, `pay_total_value`, `pay_mean_value`, `orders_count`, `items_total_qty`, and their derived short-lag/rolling features
8. **No external data sources** — all features come from the provided CSV dates or deterministic calendar transforms of those dates

This is documented in the artifact `run_audit.json` under `leakage_policy`.

---

## 5. Model Architecture

### 5.1 Pipeline A: Recursive LightGBM Ensemble

**File**: `model_thang/forecast_pipeline.py`

**Approach**: Standard recursive forecasting — train a model to predict 1-day-ahead, then feed predictions back as inputs for multi-step forecasting.

**Model specifications** (3 specs × 3 seeds = 9 models per target):

| Spec Name | Objective | Num Leaves | Target Transform |
|---|---|---|---|
| `log_l1_leaves48` | `regression_l1` (MAE) | 48 | `log1p` |
| `log_l2_leaves63` | `regression` (MSE) | 63 | `log1p` |
| `raw_tweedie_leaves40` | `tweedie` (variance=1.45) | 40 | `raw` |

**Common hyperparameters**:
- Learning rate: 0.025
- Feature fraction: 0.8 (column subsampling)
- Bagging fraction: 0.85 (row subsampling)
- L1: 0.1, L2: 1.0
- Early stopping: 150 rounds
- Final refit on full history with 1.08× best iteration

**Prediction**: 9 models predict independently, then average.

**Why multiple specs + seeds**:
- `regression_l1` is robust to outliers (directly optimizes MAE)
- `regression` (MSE) captures the overall variance structure
- `tweedie` handles zero-inflated or right-skewed distributions
- Multiple seeds reduce variance via bagging

### 5.2 Pipeline B: Direct Multi-Step Model

**File**: `model_thang/explainable_forecast_factory.py`

**Approach**: Instead of recursive 1-step prediction, train the model on **pseudo-horizon rows** where each training example mimics "forecast day D from cutoff C, horizon = D - C days".

**Training rows are constructed from 15 historical cutoffs** (2014-12-31 through 2021-07-01). For each cutoff, we create rows for the next 548 days using only information known at that cutoff.

**Components**:
1. **LightGBM**: Same lag/calendar/seasonal features, trained to predict target at (cutoff, horizon)
   - Objective: `regression_l1` (MAE)
   - Num leaves: 31, learning rate: 0.025
   - Feature fraction: 0.85
   - Early stopping: 100 rounds

2. **RidgeCV**: Linear model on the same feature set
   - Pipeline: SimpleImputer(median) → StandardScaler → RidgeCV
   - Alphas searched: logspace(-2, 4, 13)
   - Target: log1p transformed

3. **DoY Prior**: Simple day-of-year mean as a seasonal baseline

**Weighted ensemble**: Grid search (step=0.05) finds optimal weights minimizing OOF MAE. Typical weights: ~45% LGBM, ~35% Ridge, ~20% DoY prior.

**Why direct multi-step**:
- Avoids error accumulation from recursive prediction over 548 days
- Learns horizon-dependent behavior (short vs medium vs long-term forecasts differ)
- Better calibration for long horizons

### 5.3 Pipeline C: Legacy Model Versions (v1–v4)

**Files**: `src/final_model.py`, `src/final_model_v2.py`, `src/final_model_v3.py`, `src/final_model_v4.py`

Each legacy version is a **single-target recursive LightGBM** with progressively richer feature engineering:

| Version | Key Innovation | Features |
|---|---|---|
| v1 | Base recursive LGBM with target-proxy guard | Calendar + basic exogenous, excluding same-day proxy columns (~89 features) |
| v2 | Improved FE + log-linear calibration | More lags, better imputation (~80 features) |
| v3 | Hyperparameter tuning | Tweaked learning rate, leaves, regularization |
| v4 | Big FE overhaul | Full v4 exogenous module (~100+ features from all 13 CSVs) |

Each version is calibrated with `regime_recovery` levels and contributes to the M5-style blend.

### 5.4 Final Blend: M5-Style 80/20

**File**: `model_thang/build_direct_lgb_candidates.py`

The final `submission.csv` is:

```
submission = 0.80 × M5_blend + 0.20 × Direct_LGBM_regime_recovery
```

Where:
- **M5_blend** = 0.50 × v1-no-proxy + 0.30 × v2 + 0.05 × v3 + 0.15 × v4 (all regime_recovery calibrated)
- **Direct_LGBM_regime_recovery** = Direct LightGBM component from Pipeline B

**Why this blend**:
- M5-style blending (inspired by the M5 forecasting competition) combines diverse model versions to reduce variance
- The direct component adds a different inductive bias (non-recursive) that complements the recursive models
- 80/20 was selected by comparing Kaggle leaderboard feedback across multiple submissions

---

## 6. Training & Validation

### 6.1 Time-Aware Cross-Validation

**Critical rule**: In time series, standard k-fold CV leaks future information. We use **walk-forward validation** where training data always precedes validation data.

### 6.2 Walk-Forward CV Details

**Pipeline A (Recursive)**:

| Fold | Training Period | Validation Period | Horizon |
|---|---|---|---|
| fold_2020_2021 | 2014-01-01 to 2020-06-30 | 2020-07-01 to 2021-12-31 | 548 days |
| fold_2021_2022 | 2014-01-01 to 2021-06-30 | 2021-07-01 to 2022-12-31 | 548 days |

Each fold mimics the actual forecasting task: predict 548 days ahead from a cutoff.

**Pipeline B (Direct)**:

| Fold | Training Cutoffs | Validation Cutoff |
|---|---|---|
| 1 | All cutoffs < 2020-06-30 | 2020-06-30 |
| 2 | All cutoffs < 2020-12-31 | 2020-12-31 |
| 3 | All cutoffs < 2021-07-01 | 2021-07-01 |

Each validation cutoff produces a 548-day horizon pseudo-forecast using only data known at that cutoff.

**Baseline models** use per-year validation:

| Val Year | Training | Validation |
|---|---|---|
| 2020 | 2014-2019 | 2020 |
| 2021 | 2014-2020 | 2021 |
| 2022 | 2014-2021 | 2022 |

Results are averaged across folds for reporting.

---

## 7. Yearly Level Calibration

The recursive model tends to produce forecasts whose **annual mean** drifts from the expected level (due to error accumulation over 548 days). We correct this with post-hoc calibration.

**Regime Recovery Strategy** (the calibration used in the final submission):

```
The business experienced a structural break around 2019 (-40% daily revenue).
By 2022, recovery is underway. We project partial reversion toward the
pre-break baseline (2014-2018 average).

For Revenue:
  mean_2023 = actual_2022 + 0.40 × (pre_break_mean - actual_2022)
  mean_2024 = actual_2022 + 0.80 × (pre_break_mean - actual_2022)

For COGS (recovers slightly faster due to margin compression during promos):
  mean_2023 = actual_2022 + 0.55 × (pre_break_mean - actual_2022)
  mean_2024 = actual_2022 + 0.85 × (pre_break_mean - actual_2022)
```

**Alternative strategies also computed** (for ablation):
- `recent_mean`: Flat level = 2019-2022 average
- `yoy_continuation`: 2022 × (2022/2021 ratio) extrapolated
- `log_linear_2019`: Log-linear fit on 2019-2022
- `blend`: 55/45 mix of recent_mean and yoy_continuation
- `recovery_upper`: More optimistic recovery scenario

---

## 8. Explainability

### SHAP (SHapley Additive exPlanations)

Computed for the Direct LGBM model using `shap.TreeExplainer`:
- Sample size: 2000 rows from training set
- Output: Mean absolute SHAP value per feature (averaged across all samples)
- Artifacts: `direct_factory_shap_importance.csv`, `top30_features_shap_overall.png`

### Feature Importance by Gain

LightGBM's native feature importance (total gain from splits using that feature):
- Computed per model, averaged across seeds
- Artifacts: `feature_importance_gain.csv`, `direct_factory_feature_importance.csv`

### Feature Group Analysis

Features are categorized into business-meaningful groups:

| Group | Examples | Business Meaning |
|---|---|---|
| `seasonal_prior` | `doy_mean_cutoff`, `month_dow_mean_cutoff` | Expected seasonal demand pattern |
| `target_lag` | `forecast_lag365`, `anchor_lag364` | Year-over-year memory and anchoring |
| `anchor_level` | `roll_mean365`, `anchor_latest` | Recent demand level |
| `calendar_seasonality` | `month`, `dow`, `sin_year_1` | Time-based patterns |
| `holiday_event` | `days_to_tet`, `is_1111` | Holiday/event-driven spikes |
| `regime_level` | `post_to_pre_ratio`, `annual_mean_cutoff` | Structural break and recovery |
| `horizon` | `horizon`, `horizon_gt_365` | Forecast distance effects |

### Top Feature Insights

The model's top drivers of Revenue are typically:
1. **Seasonal priors** (day-of-year mean from history) — fashion demand has strong yearly recurrence
2. **Target lags** (365/730-day anchors) — year-over-year demand consistency
3. **Holiday events** (Tet, 11/11, Black Friday) — major demand disruption windows
4. **Regime level** (post-2019 recovery) — structural shift in the business
5. **Calendar features** (month, day-of-week) — within-week and within-month patterns

---

## 9. Results Comparison

See `docs/tables/baseline_results.csv` and `docs/tables/pipeline_results.csv`.

### Baseline Models (3-fold walk-forward CV average)

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Revenue | XGBoost | 140,512 | 202,028 | 0.9850 |
| Revenue | Random Forest | 197,197 | 302,103 | 0.9664 |
| Revenue | Ridge | 338,886 | 443,939 | 0.9263 |
| Revenue | Linear Regression | 346,537 | 450,251 | 0.9239 |
| Revenue | Lasso | 340,817 | 444,754 | 0.9260 |
| COGS | XGBoost | 175,309 | 256,796 | 0.9664 |
| COGS | Ridge | 246,283 | 335,014 | 0.9441 |
| COGS | Linear Regression | 246,725 | 335,526 | 0.9440 |
| COGS | Lasso | 249,116 | 338,291 | 0.9431 |
| COGS | Random Forest | 270,877 | 389,526 | 0.9239 |

> **Note**: These baselines use 1-day-ahead features (target lags from the same series) on a known validation year. The R² appears high because predicting 1 day ahead is much easier than predicting 548 days ahead recursively.

### Pipeline Models (548-day horizon walk-forward CV)

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Revenue | Direct LGBM+Ridge Ensemble | 503,012 | 733,825 | 0.7994 |
| Revenue | Direct LGBM Only | 523,655 | 775,814 | 0.7708 |
| Revenue | Recursive LGBM Ensemble (CV-weighted) | 631,776 | 879,062 | 0.6804 |
| Revenue | Recursive LGBM (9-model bag) | 663,052 | 912,973 | 0.6550 |
| Revenue | DoY Climatology Mean | 682,579 | 933,401 | 0.6395 |
| Revenue | Seasonal Naive (lag-364) | 828,618 | 1,218,866 | 0.3853 |
| COGS | Direct LGBM+Ridge Ensemble | 441,595 | 633,897 | 0.8038 |
| COGS | Direct LGBM Only | 452,564 | 659,500 | 0.7838 |
| COGS | Recursive LGBM Ensemble (CV-weighted) | 547,400 | 758,884 | 0.6876 |
| COGS | Recursive LGBM (9-model bag) | 574,962 | 791,782 | 0.6600 |
| COGS | DoY Climatology Mean | 594,280 | 818,337 | 0.6364 |
| COGS | Seasonal Naive (lag-364) | 728,543 | 1,083,031 | 0.3624 |

**Key insight**: The Direct model significantly outperforms the Recursive model on the 548-day horizon (R² 0.80 vs 0.68 for Revenue), confirming that error accumulation is a major challenge for recursive approaches.

---

## 10. File Structure

```
final_thang_model/
├── submission.csv                          ← Final Kaggle submission (548 rows)
├── train_save_infer_blend.py               ← End-to-end train/save/infer/blend entrypoint
├── README.md
│
├── scripts/
│   ├── run_baselines.py                    ← Script to generate baseline/model tables
│   ├── reproduce_submission.py             ← Validate/copy final candidate only
│   └── generate_flowchart.py               ← Regenerate docs/assets/PIPELINE_FLOWCHART.png
│
├── notebooks/
│   └── reproduce_best_kaggle_solution.ipynb
│
├── docs/
│   ├── MODEL_DOCUMENTATION.md              ← This file
│   ├── MODEL_REPORT.md
│   ├── MODEL_EXPLAINABILITY.md
│   ├── CV_DATA_SPLIT.md
│   ├── VIETNAM_HOLIDAY_FEATURE_AUDIT.md
│   ├── vietnam_calendar_events_deterministic_2012_2024.csv
│   ├── assets/
│   │   └── PIPELINE_FLOWCHART.png
│   └── tables/
│       ├── baseline_results.csv
│       ├── pipeline_results.csv
│       └── full_feature_importance.csv
│
├── src/                                    ← Core modules
│   ├── __init__.py
│   ├── final_model.py                      ← Legacy v1: base recursive LGBM with target-proxy guard
│   ├── final_model_v2.py                   ← Legacy v2: improved FE
│   ├── final_model_v3.py                   ← Legacy v3: hyperparameter tuning
│   ├── final_model_v4.py                   ← Legacy v4: big FE overhaul (~100 features)
│   ├── features_v4.py                      ← V4 feature engineering module
│   └── calendar_vn.py                      ← Deterministic Vietnamese calendar features
│
├── model_thang/                            ← Pipeline A & B implementation
│   ├── __init__.py
│   ├── forecast_pipeline.py                ← Recursive LGBM ensemble (Pipeline A)
│   ├── explainable_forecast_factory.py     ← Direct LGBM+Ridge (Pipeline B)
│   ├── build_v4_regime_candidate.py        ← V4 regime recovery calibration
│   ├── build_legacy_blend_regime.py        ← Legacy version regime calibration
│   ├── build_m5_style_blend.py             ← M5-style multi-version blend
│   ├── build_direct_lgb_candidates.py      ← Final 80/20 blend
│   ├── visualize_top_features.py           ← Feature importance visualization
│   └── artifacts/                          ← All generated artifacts
│       ├── run_audit.json                  ← Leakage policy & run metadata
│       ├── cv_metrics.csv                  ← Pipeline A CV results
│       ├── cv_weights.json                 ← Ensemble weights
│       ├── direct_factory_cv_metrics.csv   ← Pipeline B CV results
│       ├── direct_factory_shap_importance.csv
│       ├── direct_factory_feature_importance.csv
│       ├── explainable_forecast_factory_report.md
│       ├── *.csv                           ← Various submission candidates
│       └── feature_importance/             ← SHAP/gain plots
│
└── model_thang/artifacts/saved_models/     ← Saved direct LightGBM/Ridge models
```

---

## 11. How to Reproduce

```bash
# From the workspace root (vinuni_hackathon/)
cd final_thang_model

# Option 1: Run via notebook
jupyter notebook notebooks/reproduce_best_kaggle_solution.ipynb

# Option 2: Run the full train/save/infer/blend pipeline
uv run python train_save_infer_blend.py --skip-visuals

# The final submission.csv is produced by copying
# model_thang/artifacts/advanced_experiments/submission_m5_lgb_direct_blend_80_20.csv

# Run baseline comparison
PYTHONPATH=. python scripts/run_baselines.py
```

**Dependencies**: pandas, numpy, lightgbm, scikit-learn, shap, xgboost, matplotlib

**Reproducibility**: All random seeds are fixed (42, 123, 2024, 20260429). LightGBM runs with `deterministic=True`.
