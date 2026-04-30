# Pipeline dự báo `model_thang`

## Phạm vi

Thư mục này chứa toàn bộ phần mô hình dự báo cho Datathon 2026 Phần 3. Nguyên
tắc đang được áp dụng rất chặt:

- `sample_submission.csv` chỉ được dùng để biết định dạng và thứ tự ngày nếu
  thật sự cần.
- Không bao giờ đọc, scale, blend, hoặc dùng `Revenue`/`COGS` từ
  `sample_submission.csv` làm đặc trưng hay target phụ.
- Mọi hiệu chỉnh level đều lấy từ `sales.csv` và các bảng train được cung cấp.

## Các artifact chính

- `forecast_pipeline.py`: pipeline train-only độc lập, gồm calendar, sự kiện
  Việt Nam, lag/rolling, climatology của exogenous signals và recursive target
  lags.
- `build_v4_regime_candidate.py`: dùng lại feature family mạnh từ
  `src.final_model_v4`, sau đó áp dụng calibration `regime_recovery` chỉ dựa
  trên train.
- `build_legacy_blend_regime.py`: rebuild các forecast v1/v2/v3/v4 và xuất
  model-diversity blend.
- `build_m5_style_blend.py`: blend kiểu M5, ít tham số, ưu tiên model diversity
  và ghi diagnostics cho từng component.
- `explainable_forecast_factory.py`: forecast factory theo hướng dự báo trực
  tiếp theo horizon, gồm LightGBM, Ridge, đặc trưng ngày lễ từ
  `docs/vietnam_holiday_calendar_2012_2024.csv`, artifact giải thích kiểu SHAP,
  và các blend rủi ro thấp với candidate M5.
- `research/time_series_ensemble_notes.md`: ghi chú research từ các contest
  time-series Kaggle/M5/G-Research và mapping sang bài này.
- `research/forecasting_pipeline_research.md`: mapping pipeline
  Ridge/LGB/Prophet/specialist/calibration sang bài Revenue/COGS.
- `artifacts/explainable_forecast_factory_report.md`: báo cáo kỹ thuật ngắn cho
  model explainability.
- `artifacts/direct_factory_feature_importance.csv`: độ quan trọng đặc trưng
  theo LightGBM gain.
- `artifacts/direct_factory_shap_importance.csv`: độ quan trọng đặc trưng theo
  mean absolute SHAP.
- `artifacts/direct_factory_cv_metrics.csv`: validation theo chiều thời gian.
- `artifacts/direct_factory_audit.json`: audit leakage, files, score và
  candidate khuyến nghị.

## Điểm Kaggle hiện tại

Đã submit các ngày 2026-04-29 và 2026-04-30:

| Tệp submit | Public score |
| --- | ---: |
| `submission_cv_ensemble_recovery_upper.csv` | 982,747.45166 |
| `submission_cv_ensemble_regime_recovery.csv` | 814,931.80478 |
| `submission_model_regime_recovery.csv` | 784,288.66083 |
| `submission_v4_regime_recovery.csv` | 761,534.82837 |
| `m5b50300515.csv` | 738,646.02933 |
| `submission_m5_direct_blend_95_05.csv` | 737,320.56562 |
| `submission_m5_direct_blend_90_10.csv` | 736,406.67678 |
| `submission_m5_direct_blend_85_15.csv` | 735,769.48625 |
| `submission_m5_lgb_direct_blend_85_15.csv` | 734,525.39095 |
| `submission_m5_lgb_direct_blend_80_20.csv` | 734,211.35297 |
| `submission_m5_lgb_direct_blend_75_25.csv` | 734,364.08799 |
| `submission_m5_lgb_direct_blend_79_21.csv` | 734,222.62152 |

Candidate tốt nhất đã submit trong thư mục này là:

```text
artifacts/advanced_experiments/submission_m5_lgb_direct_blend_80_20.csv
```

File này blend 80% candidate M5-style v1/v2/v3/v4 với 20% thành phần
LightGBM-only dự báo trực tiếp theo horizon. Biến thể này được tạo sau validation
548 ngày `2021-01-01` đến `2022-07-02`, trong đó direct LightGBM có MAE tốt hơn
direct weighted blend trên Revenue. Probe 75/25 và 79/21 không vượt 80/20.

## Cách tính metric

### Kaggle publicScore

Kaggle API metadata của competition ghi `evaluationMetric = Mean Absolute
Error`. Vì vậy số `publicScore` trên leaderboard là **MAE**, và càng thấp càng
tốt.

Với file submit có hai cột dự báo `Revenue` và `COGS`, cách hiểu thực tế là
Kaggle tính sai số tuyệt đối trung bình trên các giá trị dự báo so với hidden
ground truth:

```text
MAE = mean(abs(y_true - y_pred))
```

Nếu tính tách theo hai target rồi lấy trung bình, công thức tương đương là:

```text
MAE_public = (MAE_Revenue + MAE_COGS) / 2
```

do hai target có cùng số dòng ngày dự báo. Public leaderboard chỉ dùng phần
public split của hidden test; private score có thể khác sau khi ban tổ chức chấm
phần còn lại.

### Metrics trong validation nội bộ

Trong pipeline nội bộ, mình vẫn báo cáo đủ ba metrics theo đề:

- **MAE**: trung bình sai số tuyệt đối. Càng thấp càng tốt.

```text
MAE = (1 / n) * sum(|actual_i - pred_i|)
```

- **RMSE**: căn bậc hai của trung bình bình phương sai số. Metric này phạt nặng
  các ngày forecast sai lớn.

```text
RMSE = sqrt((1 / n) * sum((actual_i - pred_i)^2))
```

- **R2**: tỷ lệ phương sai được mô hình giải thích. Càng gần 1 càng tốt; âm
  nghĩa là model tệ hơn dự báo bằng trung bình.

```text
R2 = 1 - sum((actual_i - pred_i)^2) / sum((actual_i - mean(actual))^2)
```

Tóm lại: **Kaggle rank đang tối ưu MAE**, còn **RMSE và R2 nên đưa vào báo cáo
kỹ thuật** để chứng minh mô hình ổn định, không chỉ ăn điểm trên một scalar.

## Cách reproduce

Chạy từ repo root:

```bash
uv run python model_thang/forecast_pipeline.py
uv run python model_thang/build_v4_regime_candidate.py
uv run python model_thang/build_legacy_blend_regime.py
uv run python model_thang/build_m5_style_blend.py
uv run python model_thang/explainable_forecast_factory.py
```

Submit candidate tốt nhất đã có điểm:

```bash
uv run kaggle competitions submit \
  -c datathon-2026-round-1 \
  -f model_thang/artifacts/advanced_experiments/submission_m5_lgb_direct_blend_80_20.csv \
  -m "model_thang m5 lgb direct blend 80 20 validation 2021"
```

Candidate chỉ nên thử nếu muốn kiểm tra sâu hơn vùng weight cao hơn, vì 75/25
đã xấu hơn 80/20:

```bash
uv run kaggle competitions submit \
  -c datathon-2026-round-1 \
  -f model_thang/artifacts/advanced_experiments/submission_m5_lgb_direct_blend_70_30.csv \
  -m "model_thang m5 lgb direct blend 70 30 validation 2021"
```

## Chiến lược feature

- **Target lags**: 7, 14, 28, 56, 91, 182, 364, 365, 371, 548, 728, 730 ngày.
- **Rolling target features**: rolling mean/std/median theo các cửa sổ 7, 14,
  28, 56, 91, 182, 365 ngày.
- **Calendar**: month/day/dow/week/quarter, Fourier weekly/yearly, payday
  windows.
- **Vietnam holiday/e-commerce events**: Tet, Hung Kings, 30/4, 1/5, 2/9,
  Mid-Autumn, 11/11, Black Friday, Cyber Monday, 12/12, Valentine, 8/3, 20/10,
  20/11, Christmas.
- **Exogenous features**: orders, item/product mix, payments, web traffic,
  returns, reviews, shipments, inventory, promotion và customer lifecycle.
- **Exogenous an toàn cho forecast**: do tương lai không có operational tables thật,
  exogenous được biểu diễn bằng lag hoặc day-of-year climatology tính từ train
  cutoff.
- **Factory dự báo trực tiếp theo horizon**: các historical pseudo-cutoffs dự báo 548 ngày
  tiếp theo, chỉ dùng thông tin biết được tại cutoff. OOF CV chọn final direct
  weights là 90% LightGBM và 10% day-of-year prior cho cả `Revenue` và `COGS`.

## Calibration level

`regime_recovery` là train-only:

- Phát hiện structural break năm 2019 từ `sales.csv`.
- Dùng trung bình full-year 2014-2018 làm pre-break baseline.
- Bắt đầu từ mean năm 2022.
- Project phục hồi một phần về baseline:
  - Revenue: phục hồi 40% gap trong 2023, 80% gap trong 2024.
  - COGS: phục hồi 55% gap trong 2023, 85% gap trong 2024.

Không có target từ sample submission hoặc public leaderboard được dùng trong
công thức calibration này.

## Explainability cho báo cáo kỹ thuật

Các artifact nên đưa vào report:

- `artifacts/explainable_forecast_factory_report.md`
- `artifacts/direct_factory_cv_metrics.csv`
- `artifacts/direct_factory_feature_group_importance.csv`
- `artifacts/direct_factory_shap_group_importance.csv`
- `artifacts/direct_factory_shap_importance.csv`

Kết luận business chính từ model:

- Calendar/seasonality là driver lớn nhất vì fashion demand có mùa vụ năm rất
  mạnh.
- Target lags, đặc biệt yearly memory 364/365/728/730 ngày, giữ forecast bám
  vào shape lịch sử.
- Holiday/event features có tín hiệu rõ: Tet, Hung Kings, 30/4-1/5, 11/11,
  Black Friday, 12/12 và các ngày gifting.
- Regime features cần thiết vì dữ liệu có break lớn năm 2019 và phục hồi năm
  2022.
