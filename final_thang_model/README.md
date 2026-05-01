# Final Thang Model

Package này là bản nộp cuối cho phần modeling. Nó tự chạy lại được khi có:

```text
vinuni_hackathon/
  data/
  final_thang_model/
```

## File chính

- `reproduce_best_kaggle_solution.ipynb`: notebook reproduce theo flow data science, từ audit dữ liệu, EDA ngắn, chạy pipeline local, validate và export submission.
- `train_save_infer_blend.py`: script end-to-end train model, lưu model, load model để inference, blend và export `submission.csv`.
- `reproduce_submission.py`: script kiểm tra/copy final candidate hiện có ra `submission.csv` khi không cần train lại.
- `submission.csv`: file submit cuối, sinh từ candidate local `submission_m5_lgb_direct_blend_80_20.csv`.
- `MODEL_REPORT.md`: báo cáo kỹ thuật tiếng Việt về feature engineering, model, validation, leakage guard và explainability.
- `model_thang/`: các script pipeline đã được copy vào package để không phụ thuộc folder ngoài.
- `src/`: module feature/model local dùng bởi pipeline.
- `docs/`: CSV calendar deterministic sinh từ dương lịch và audit feature holiday dùng bởi model explainable.

## Chạy lại

Mở notebook và chạy top-to-bottom từ repo root hoặc từ chính folder `final_thang_model`.

Notebook sẽ chạy các script local:

```text
model_thang/forecast_pipeline.py
model_thang/build_v4_regime_candidate.py
model_thang/build_legacy_blend_regime.py
model_thang/build_m5_style_blend.py
model_thang/explainable_forecast_factory.py
model_thang/build_direct_lgb_candidates.py
model_thang/visualize_top_features.py
```

Hoặc chạy toàn bộ bằng Python script:

```bash
cd final_thang_model
uv run python train_save_infer_blend.py
```

Direct model được lưu tại:

```text
model_thang/artifacts/saved_models/direct_factory/
```

Direct inference từ saved model được ghi tại:

```text
model_thang/artifacts/inference/
```

Các script đã được chỉnh path để đọc `../data` và ghi artifact vào
`final_thang_model/model_thang/artifacts`.

## CV metrics

Các kết quả CV lưu theo từng fold/cutoff và có thêm bảng tổng hợp mean/std:

```text
model_thang/artifacts/cv_metrics.csv
model_thang/artifacts/direct_factory_cv_metrics.csv
model_thang/artifacts/cv_metrics_mean_std.csv
model_thang/artifacts/direct_factory_cv_metrics_mean_std.csv
model_thang/artifacts/model_baseline_metrics_mean_std.csv
```

`model_baseline_metrics_mean_std.csv` gộp recursive pipeline CV, direct factory
CV và baseline single-split. Các baseline single-split có `n_folds = 1` nên cột
`*_std` để trống.

## Submit

Submission đã test Kaggle:

```text
publicScore: 730,879.20779
description: train-save-infer-blend deterministic calendar pipeline
```

Submit lại bằng:

```bash
uv run kaggle competitions submit \
  -c datathon-2026-round-1 \
  -f final_thang_model/submission.csv \
  -m "train-save-infer-blend deterministic calendar pipeline"
```

## Leakage policy

- Không dùng `Revenue/COGS` từ `sample_submission.csv`.
- Không scale/blend theo sample target.
- Future features chỉ dùng calendar/holiday deterministic sinh từ `Date`, lag/seasonal prior hợp lệ hoặc prediction đã sinh trong recursive inference.
- Validation và direct-horizon rows kiểm tra cutoff để tránh dùng actual future lag.
