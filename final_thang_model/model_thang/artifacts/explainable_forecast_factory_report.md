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
| Revenue  | 2020-06-30    | lgb             | 524637           | 796969           |  0.7372 |
| Revenue  | 2020-06-30    | ridge           |      2.49304e+06 |      2.89121e+06 | -2.459  |
| Revenue  | 2020-06-30    | doy_prior       |      1.51774e+06 |      1.83397e+06 | -0.3918 |
| Revenue  | 2020-12-31    | lgb             | 625660           | 901193           |  0.7389 |
| Revenue  | 2020-12-31    | ridge           |      2.8814e+06  |      3.31066e+06 | -2.524  |
| Revenue  | 2020-12-31    | doy_prior       |      1.30527e+06 |      1.68428e+06 |  0.0879 |
| Revenue  | 2021-07-01    | lgb             | 438916           | 664956           |  0.8167 |
| Revenue  | 2021-07-01    | ridge           | 669245           | 941294           |  0.6326 |
| Revenue  | 2021-07-01    | doy_prior       | 899250           |      1.19517e+06 |  0.4077 |
| Revenue  | all_oof       | weighted_direct | 504399           | 742145           |  0.7948 |
| COGS     | 2020-06-30    | lgb             | 428369           | 637186           |  0.7756 |
| COGS     | 2020-06-30    | ridge           |      2.01115e+06 |      2.33148e+06 | -2.0038 |
| COGS     | 2020-06-30    | doy_prior       |      1.29019e+06 |      1.55701e+06 | -0.3396 |
| COGS     | 2020-12-31    | lgb             | 543098           | 771555           |  0.7503 |
| COGS     | 2020-12-31    | ridge           |      2.3049e+06  |      2.65733e+06 | -1.9614 |
| COGS     | 2020-12-31    | doy_prior       |      1.0272e+06  |      1.33831e+06 |  0.2489 |
| COGS     | 2021-07-01    | lgb             | 385504           | 565814           |  0.8288 |
| COGS     | 2021-07-01    | ridge           | 546033           | 742003           |  0.7055 |
| COGS     | 2021-07-01    | doy_prior       | 756748           |      1.00994e+06 |  0.4544 |
| COGS     | all_oof       | weighted_direct | 440726           | 633629           |  0.804  |

## Direct Component Weights

| target   |   lgb |   ridge |   doy_prior |
|:---------|------:|--------:|------------:|
| Revenue  |   0.9 |       0 |         0.1 |
| COGS     |   0.9 |       0 |         0.1 |

## Top Feature Groups By LightGBM Gain

| target   | group                |       gain |
|:---------|:---------------------|-----------:|
| COGS     | calendar_seasonality | 605393     |
| COGS     | target_lag           | 238143     |
| COGS     | holiday_event        | 104964     |
| COGS     | anchor_level         |  11863.8   |
| COGS     | regime_level         |  11793.4   |
| COGS     | horizon              |   6899.14  |
| COGS     | other                |    324.966 |
| Revenue  | calendar_seasonality | 584945     |
| Revenue  | target_lag           | 263412     |
| Revenue  | holiday_event        | 117373     |
| Revenue  | anchor_level         |  10616.5   |
| Revenue  | regime_level         |  10354.2   |
| Revenue  | horizon              |   6542.76  |
| Revenue  | other                |    454.249 |

## Top Feature Groups By SHAP

| target   | group                |   mean_abs_shap |
|:---------|:---------------------|----------------:|
| COGS     | calendar_seasonality |        0.612165 |
| COGS     | target_lag           |        0.160663 |
| COGS     | holiday_event        |        0.111717 |
| COGS     | regime_level         |        0.027691 |
| COGS     | anchor_level         |        0.008111 |
| COGS     | horizon              |        0.001663 |
| COGS     | other                |        0.000136 |
| Revenue  | calendar_seasonality |        0.573752 |
| Revenue  | target_lag           |        0.221546 |
| Revenue  | holiday_event        |        0.111961 |
| Revenue  | regime_level         |        0.019209 |
| Revenue  | anchor_level         |        0.009369 |
| Revenue  | horizon              |        0.001244 |
| Revenue  | other                |        0.000396 |

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
