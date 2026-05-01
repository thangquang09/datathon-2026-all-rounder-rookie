# Sales Forecast Submission

Package này là bản nộp cuối cho phần modeling. Nó tự chạy lại được khi có:

```text
vinuni_hackathon/
  data/
  sales_forecast_submission/
```

## File chính

- `train_save_infer_blend.py`: script end-to-end train model, lưu model, load model để inference, blend và export `submission.csv`.
- `submission.csv`: file submit cuối, sinh từ candidate local `submission_m5_lgb_direct_blend_80_20.csv`.
- `notebooks/reproduce_best_kaggle_solution.ipynb`: notebook reproduce theo flow data science, từ audit dữ liệu, EDA ngắn, chạy pipeline local, validate và export submission.
- `scripts/reproduce_submission.py`: script kiểm tra/copy final candidate hiện có ra `submission.csv` khi không cần train lại.
- `scripts/run_baselines.py`: script sinh lại bảng baseline/model comparison trong `docs/tables/`.
- `docs/MODEL_REPORT.md`: báo cáo kỹ thuật tiếng Việt về feature engineering, model, validation, leakage guard và explainability.
- `docs/CV_DATA_SPLIT.md`: mô tả cách chia train/validation, walk-forward CV, direct pseudo-cutoff CV và artifact mean/std.
- `pipeline/`: các script pipeline đã được copy vào package để không phụ thuộc folder ngoài.
- `legacy_components/`: module feature/model local dùng bởi pipeline.
- `artifacts/`: model outputs, saved models, metrics, audit files, final candidates và README giải thích từng nhóm artifact.
- `docs/`: report, deterministic calendar CSV, audit holiday feature, figure assets và result tables.

## Chạy lại

Mở notebook và chạy top-to-bottom từ repo root hoặc từ chính folder `sales_forecast_submission`.

```text
notebooks/reproduce_best_kaggle_solution.ipynb
```

Notebook sẽ chạy các script local:

```text
pipeline/forecast_pipeline.py
pipeline/build_v4_regime_candidate.py
pipeline/build_legacy_blend_regime.py
pipeline/build_m5_style_blend.py
pipeline/explainable_forecast_factory.py
pipeline/build_direct_lgb_candidates.py
pipeline/visualize_top_features.py
```

Hoặc chạy toàn bộ bằng Python script:

```bash
cd sales_forecast_submission
uv run python train_save_infer_blend.py
```

Nếu chỉ muốn kiểm tra/copy final candidate đã sinh:

```bash
cd sales_forecast_submission
uv run python scripts/reproduce_submission.py --overwrite
```

Direct model được lưu tại:

```text
artifacts/saved_models/direct_factory/
```

Direct inference từ saved model được ghi tại:

```text
artifacts/inference/
```

Các script đã được chỉnh path để đọc `../data` và ghi artifact vào
`sales_forecast_submission/artifacts`.

## CV metrics

Các kết quả CV lưu theo từng fold/cutoff và có thêm bảng tổng hợp mean/std:

```text
artifacts/cv_metrics.csv
artifacts/direct_factory_cv_metrics.csv
artifacts/cv_metrics_mean_std.csv
artifacts/direct_factory_cv_metrics_mean_std.csv
artifacts/model_baseline_metrics_mean_std.csv
```

`model_baseline_metrics_mean_std.csv` gộp recursive pipeline CV, direct factory
CV và baseline single-split. Các baseline single-split có `n_folds = 1` nên cột
`*_std` để trống.

## Submit

Submission đã test Kaggle:

```text
publicScore: 730,067.90380
description: ablation drop v1 target proxy features
```

Submit lại bằng:

```bash
uv run kaggle competitions submit \
  -c datathon-2026-round-1 \
  -f sales_forecast_submission/submission.csv \
  -m "ablation drop v1 target proxy features"
```

## Leakage policy

- Không dùng `Revenue/COGS` từ `sample_submission.csv`.
- Không scale/blend theo sample target.
- Legacy `v1` vẫn nằm trong M5 blend nhưng đã loại các target-proxy same-day
  (`items_gross_value`, `pay_total_value`, `pay_mean_value`, `orders_count`,
  `items_total_qty` và các lag/rolling dẫn xuất từ chúng).
- Future features chỉ dùng calendar/holiday deterministic sinh từ `Date`, lag/seasonal prior hợp lệ hoặc prediction đã sinh trong recursive inference.
- Validation và direct-horizon rows kiểm tra cutoff để tránh dùng actual future lag.
