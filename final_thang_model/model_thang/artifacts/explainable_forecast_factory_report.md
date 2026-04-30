# Explainable Forecast Factory Report

## Objective

Forecast daily `Revenue` and `COGS` for 2023-01-01 to 2024-07-01 with a reproducible, leakage-safe pipeline.

## Scoring Alignment

- Leaderboard: produces Kaggle-ready submissions and blends against the current best no-sample candidate.
- Technical report: uses time-aware CV, explicit holiday audit features, LightGBM gain importance, SHAP mean absolute importance, and feature-group business explanations.

## Leakage Policy

- Does not read target values from `sample_submission.csv`.
- Future dates use public calendar features, historical target lags known at each cutoff, and train-cutoff seasonal priors only.
- Pseudo-horizon rows mimic historical forecast cutoffs rather than using same-day future operational aggregates.
- Yearly calibration uses only `sales.csv` historical regime assumptions.

## Time-Aware CV

| target   | fold_cutoff   | model           |              mae |             rmse |      r2 |
|:---------|:--------------|:----------------|-----------------:|-----------------:|--------:|
| Revenue  | 2020-06-30    | lgb             | 528581           | 800948           |  0.7345 |
| Revenue  | 2020-06-30    | ridge           |      2.47978e+06 |      2.87676e+06 | -2.4245 |
| Revenue  | 2020-06-30    | doy_prior       |      1.51774e+06 |      1.83397e+06 | -0.3918 |
| Revenue  | 2020-12-31    | lgb             | 601202           | 861855           |  0.7612 |
| Revenue  | 2020-12-31    | ridge           |      2.84904e+06 |      3.27444e+06 | -2.4474 |
| Revenue  | 2020-12-31    | doy_prior       |      1.30527e+06 |      1.68428e+06 |  0.0879 |
| Revenue  | 2021-07-01    | lgb             | 441182           | 664638           |  0.8168 |
| Revenue  | 2021-07-01    | ridge           | 677453           | 961442           |  0.6167 |
| Revenue  | 2021-07-01    | doy_prior       | 899250           |      1.19517e+06 |  0.4077 |
| Revenue  | all_oof       | weighted_direct | 503012           | 733825           |  0.7994 |
| COGS     | 2020-06-30    | lgb             | 435557           | 650611           |  0.7661 |
| COGS     | 2020-06-30    | ridge           |      1.99445e+06 |      2.31364e+06 | -1.958  |
| COGS     | 2020-06-30    | doy_prior       |      1.29019e+06 |      1.55701e+06 | -0.3396 |
| COGS     | 2020-12-31    | lgb             | 535902           | 761583           |  0.7568 |
| COGS     | 2020-12-31    | ridge           |      2.26226e+06 |      2.61075e+06 | -1.8584 |
| COGS     | 2020-12-31    | doy_prior       |      1.0272e+06  |      1.33831e+06 |  0.2489 |
| COGS     | 2021-07-01    | lgb             | 386235           | 566307           |  0.8285 |
| COGS     | 2021-07-01    | ridge           | 544513           | 744370           |  0.7036 |
| COGS     | 2021-07-01    | doy_prior       | 756748           |      1.00994e+06 |  0.4544 |
| COGS     | all_oof       | weighted_direct | 441595           | 633897           |  0.8038 |

## Direct Component Weights

| target   |   lgb |   ridge |   doy_prior |
|:---------|------:|--------:|------------:|
| Revenue  |   0.9 |       0 |         0.1 |
| COGS     |   0.9 |       0 |         0.1 |

## Top Feature Groups By LightGBM Gain

| target   | group                |       gain |
|:---------|:---------------------|-----------:|
| COGS     | calendar_seasonality | 596431     |
| COGS     | target_lag           | 237498     |
| COGS     | holiday_event        | 100588     |
| COGS     | regime_level         |  13944.5   |
| COGS     | anchor_level         |  13158.5   |
| COGS     | horizon              |   5762.27  |
| COGS     | other                |    396.587 |
| Revenue  | calendar_seasonality | 578926     |
| Revenue  | target_lag           | 266505     |
| Revenue  | holiday_event        | 112893     |
| Revenue  | anchor_level         |  12992.5   |
| Revenue  | regime_level         |  11461     |
| Revenue  | horizon              |   5562.42  |
| Revenue  | other                |    387.471 |

## Top Feature Groups By SHAP

| target   | group                |   mean_abs_shap |
|:---------|:---------------------|----------------:|
| COGS     | calendar_seasonality |        0.609275 |
| COGS     | target_lag           |        0.163764 |
| COGS     | holiday_event        |        0.099235 |
| COGS     | regime_level         |        0.031536 |
| COGS     | anchor_level         |        0.013378 |
| COGS     | horizon              |        0.001797 |
| COGS     | other                |        0.0001   |
| Revenue  | calendar_seasonality |        0.57763  |
| Revenue  | target_lag           |        0.224136 |
| Revenue  | holiday_event        |        0.100945 |
| Revenue  | regime_level         |        0.022215 |
| Revenue  | anchor_level         |        0.014591 |
| Revenue  | horizon              |        0.001483 |
| Revenue  | other                |        0.000231 |

## Business Explanation

- `seasonal_prior`: the model relies on day-of-year/month-weekday priors because fashion demand has stable annual shape and the data generator preserves strong yearly recurrence.
- `target_lag` and `anchor_level`: recent known demand level and one-/two-year memory anchor the forecast to the latest business regime.
- `holiday_event`: Tet, Hung Kings, Apr 30-May 1, 11/11, Black Friday, 12/12 and gifting days identify demand or logistics disruption windows.
- `regime_level`: post-2019 structural break and 2022 recovery are modeled explicitly before calibration.
- `horizon`: the direct model learns different behavior for short, medium, and >365-day forecasts, reducing recursive drift.

## Generated Files

- `direct_raw`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_direct_factory_raw.csv`
- `direct_regime`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_direct_factory_regime_recovery.csv`
- `component_debug`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/direct_factory_component_debug.csv`
- `m5_direct_95_05`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_m5_direct_blend_95_05.csv`
- `m5_direct_90_10`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_m5_direct_blend_90_10.csv`
- `m5_direct_85_15`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_m5_direct_blend_85_15.csv`
- `m5_direct_80_20`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_m5_direct_blend_80_20.csv`
- `m5_direct_75_25`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_m5_direct_blend_75_25.csv`
- `m5_direct_70_30`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/submission_m5_direct_blend_70_30.csv`
- `cv_metrics`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/direct_factory_cv_metrics.csv`
- `feature_importance`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/direct_factory_feature_importance.csv`
- `shap_importance`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/direct_factory_shap_importance.csv`
- `feature_group_importance`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/direct_factory_feature_group_importance.csv`
- `audit`: `/home/thangquang09/code/vinuni_hackathon/final_thang_model/model_thang/artifacts/direct_factory_audit.json`
