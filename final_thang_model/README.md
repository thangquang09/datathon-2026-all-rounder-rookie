# Final Thang Model

Package này là bản nộp cuối cho phần modeling. Nó tự chạy lại được khi có:

```text
vinuni_hackathon/
  data/
  final_thang_model/
```

## File chính

- `reproduce_best_kaggle_solution.ipynb`: notebook reproduce theo flow data science, từ audit dữ liệu, EDA ngắn, chạy pipeline local, validate và export submission.
- `submission.csv`: file submit cuối, sinh từ candidate local `submission_m5_lgb_direct_blend_80_20.csv`.
- `MODEL_REPORT.md`: báo cáo kỹ thuật tiếng Việt về feature engineering, model, validation, leakage guard và explainability.
- `model_thang/`: các script pipeline đã được copy vào package để không phụ thuộc folder ngoài.
- `src/`: module feature/model local dùng bởi pipeline.
- `docs/`: lịch ngày lễ Việt Nam và audit feature holiday dùng bởi model explainable.

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

Các script đã được chỉnh path để đọc `../data` và ghi artifact vào
`final_thang_model/model_thang/artifacts`.

## Submit

Submission đã test Kaggle:

```text
publicScore: 733,976.28778
description: final_thang_model local multi-pipeline 80/20 M5 direct
```

Submit lại bằng:

```bash
uv run kaggle competitions submit \
  -c datathon-2026-round-1 \
  -f final_thang_model/submission.csv \
  -m "final_thang_model local multi-pipeline 80/20 M5 direct"
```

## Leakage policy

- Không dùng `Revenue/COGS` từ `sample_submission.csv`.
- Không scale/blend theo sample target.
- Future features chỉ dùng calendar, holiday, lag/seasonal prior hợp lệ hoặc prediction đã sinh trong recursive inference.
- Validation và direct-horizon rows kiểm tra cutoff để tránh dùng actual future lag.
