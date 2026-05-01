# Artifacts Guide

This directory contains generated outputs from `train_save_infer_blend.py`.
Most CSV files are intentionally kept for auditability, not because every file
is a Kaggle submission candidate.

## Final Submission Path

- `final_candidates/submission_m5_lgb_direct_blend_80_20.csv`: locked final
  local candidate copied to `../submission.csv`.
- `../submission.csv`: file submitted to Kaggle.

## Reproduction Manifest

- `train_save_infer_blend_manifest.json`: end-to-end run manifest with key
  output paths.
- `manifest.json`: recursive pipeline artifact manifest.
- `run_audit.json`: recursive pipeline leakage policy and run metadata.

## Metrics

- `cv_metrics.csv`: fold-level recursive pipeline CV metrics.
- `cv_metrics_mean_std.csv`: recursive pipeline mean/std summary.
- `direct_factory_cv_metrics.csv`: direct LightGBM/Ridge cutoff CV metrics.
- `direct_factory_cv_metrics_mean_std.csv`: direct factory mean/std summary.
- `model_baseline_metrics_mean_std.csv`: combined baseline/model mean/std table.
- `cv_weights.json`: recursive ensemble weights selected from CV.

## Direct Factory Artifacts

- `direct_factory_component_debug.csv`: per-date component predictions.
- `direct_factory_audit.json`: direct factory leakage audit and model metadata.
- `direct_factory_feature_importance.csv`: LightGBM gain importance.
- `direct_factory_shap_importance.csv`: SHAP importance.
- `direct_factory_feature_group_importance.csv`: grouped gain importance.
- `direct_factory_shap_group_importance.csv`: grouped SHAP importance.
- `explainable_forecast_factory_report.md`: generated explainability report.

## Saved Models And Inference

- `saved_models/direct_factory/`: saved LightGBM and Ridge models by target.
- `inference/`: outputs produced by loading saved models, without retraining.

## Figures

- `figures/`: top-30 feature importance PNGs and shortlist CSV.
- `feature_importance_gain.csv`: recursive pipeline gain summary.

## Intermediate Candidates

The following files are intermediate or ablation candidates that document how
the final blend was selected:

- `legacy_component_outputs/`: legacy model side outputs such as feature
  importance CSVs generated while building component forecasts.
- `legacy_v*_raw.csv` and `legacy_v*_regime_recovery.csv`: legacy component
  predictions.
- `legacy_blend_*.csv`: legacy-only blend probes.
- `m5blend_*.csv` and `m5b50300515.csv`: M5-style diversity blend probes.
- `submission_cv_ensemble_*.csv`, `submission_model_*.csv`,
  `submission_shape_*.csv`, `submission_direct_factory_*.csv`,
  `submission_m5_direct_blend_*.csv`: component and direct-blend candidates.
- `final_candidates/submission_m5_lgb_direct_blend_*.csv`: final-stage
  M5/direct blend ratio probes.

Do not delete intermediate candidates unless the corresponding pipeline code is
also updated, because downstream blend scripts read several of them during a
fresh reproduction run.
