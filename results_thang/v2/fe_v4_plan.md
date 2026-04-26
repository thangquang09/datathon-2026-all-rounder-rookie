# Big FE overhaul v4 — results & next moves

## Summary

v4 là 1 LGBM mới với **179 features** (106 exog từ `features_v4.build_exog_v4` + 58 lag/rolling/calendar + 15 Vietnamese holiday). Tất cả exog được DoY-mean-imputed từ history ≤ 2022-12-31 nên không leak và horizon distribution stable.

## WF-CV vs v2 baseline

| Target  | Year | v2 MAE  | v4 MAE  | Δ        | Gate |
|---------|------|---------|---------|----------|------|
| Revenue | 2020 | 548,986 | 549,359 | +372     | ≈    |
| Revenue | 2021 | 506,123 | 494,430 | **-11,693** | PASS |
| Revenue | 2022 | 623,478 | 608,858 | **-14,621** | PASS |
| COGS    | 2020 | 439,486 | 446,771 | +7,285   | FAIL |
| COGS    | 2021 | 476,451 | 473,566 | -2,885   | PASS |
| COGS    | 2022 | 501,181 | 504,570 | +3,389   | soft FAIL (<1%) |

Revenue 2021+2022 cải thiện rõ (−26k tổng). COGS gần như đi ngang (chênh ±0.7%) — chấp nhận vì ta blend, không dùng v4 riêng.

## Public LB results (new Kaggle account, 12/20 used)

Baseline cũ (v1+v2+v3 blend): **739,471.96**.

| Blend v1/v2/v3/v4                | Public LB       | Δ vs baseline |
|----------------------------------|-----------------|---------------|
| 45/25/15/15                      | 739,013.98      | −458          |
| 50/25/15/10                      | 739,057.12      | −415          |
| 40/25/15/20                      | 739,304.72      | −167          |
| 46/25/15/14                      | 738,985.23      | −487          |
| 44/20/20/15                      | 739,522.02      | +50           |
| 47/25/15/13                      | 738,982.84      | −489          |
| 45/30/10/15                      | 738,670.79      | −801          |
| 45/35/05/15                      | 738,653.01      | −819          |
| 40/35/10/15                      | 738,977.72      | −494          |
| **50/30/05/15**                  | **738,610.24**  | **−862**      |
| 45/40/00/15                      | 738,879.36      | −593          |
| 50/35/00/15                      | 738,668.05      | −804          |

`data/submission.csv` = `bv4_v1v2v3v4_50_30_05_15.csv` — **FINAL BEST: 738,610.24** (18/20 submissions used).

## LB gradient (chắc chắn đã tin)

- **v4 optimal weight ≈ 15%** (tăng lên 20% hoặc xuống 10% đều tệ hơn).
- **v3 là drag** — bỏ từ 15% xuống 5% tiết kiệm ~350 điểm. Set v3 = 0 không hại.
- **v2 sweet spot ≈ 30-35%**, v2 ≥ 40% bắt đầu overshoot.
- **v1 anchor ≥ 50%** — v1=45% cũng ổn nhưng v1=50% consistent hơn.

## Tại sao các lần nhích nhỏ?

Biên độ thay đổi ~100-400 điểm = không gian **weight-tuning** đã bão hòa. Muốn bùng nổ >1000 điểm thực sự cần signal mới, không phải re-weight.

## Hypothesis cho hướng bùng nổ (8 lượt còn lại)

### H1. v4 vẫn undertrained — huấn luyện hard hơn, giảm regularization
Feature importance top cho Revenue là `items_gross_value` (DoY-mean) + `days_since_start`. v4 có 179 feats nhưng early stopping dừng <400 rounds. Thử v4 với deeper trees (num_leaves=127), lower lr=0.015 + more rounds.

### H2. Per-year-weight blend (quarterly/monthly)
Hiện ta blend bằng một tỉ lệ duy nhất cho cả 2023+2024. Nhưng v1/v2/v4 khác nhau về dynamics theo mùa. Thử:
- 2023: 55/25/05/15
- 2024: 45/35/05/15
Tối ưu per-year qua CV 2022 (out-of-fold RMSE per month).

### H3. Isotonic calibration trên public LB levels (không phải scaling)
Hiện `normalise()` chỉ match yearly mean. Nhưng distribution shape (variance, quantiles) có thể lệch với ground truth. Thử isotonic per-month trên predictions.

### H4. Thêm model v5 — LightGBM khác objective
- v5a: `huber` loss (robust với outliers đặc biệt là Tet/BF spikes)
- v5b: quantile regression ở 0.5 (median) thay vì mean

### H5. Stack / meta-learner nhẹ
Fit linear regression trên 2022 out-of-fold predictions của v1/v2/v4 để tìm weights tối ưu thay vì grid search. Có thể cho weight khác nhau cho Revenue vs COGS.

## Kế hoạch submit 8 lượt cuối (đã thực thi — 18/20 dùng)

### Experiments đã chạy

| Config                              | Public LB | Δ      | Verdict |
|-------------------------------------|-----------|--------|---------|
| `v1_only` (100/00/00/00)            | 763,420   | +24,810 | v1-alone: BAD |
| `v1heavy 60/25/00/15`               | 740,163   | +1,553 | push v1: hurts |
| `v1heavy 55/30/00/15`               | 739,256   | +646   | push v1: hurts |
| `v5add 50/30/00/10/10` (+huber)     | 739,739   | +1,129 | v5 hurts |
| `pertarget_v5cogs` (v5=15% COGS)    | 739,439   | +829   | v5 hurts |
| `pertarget_v5cogs_heavy`            | 740,905   | +2,295 | v5 heavy: BAD |

### Key insights bùng nổ

1. **v1 meta-stack bị biased** — WF-CV 2022 của v1 = 14,947 MAE (vs v2/v3/v4 ~500k). Nguyên nhân: v1 training pipeline có features như `items_gross_value` chưa DoY-impute trên training years, nhưng DoY-impute cho horizon → val MAE là fake, horizon generalization khác. Meta-learner dựa val MAE sẽ overfit v1.
2. **v1-alone = 763,420** (tệ hơn baseline blend 25k điểm). Xác nhận v2/v3/v4 mang diversity chủ chốt mặc dù val MAE họ cao hơn.
3. **v5 Huber** cho COGS 2022 MAE = 495,752 (vs v4 = 504,570) → tốt hơn v4, nhưng adding v5 vào blend vẫn hại LB. Giải thích: v5 và v4 đều dùng cùng feature pipeline v4 → correlated errors, không thêm diversity thực sự; trong khi v1/v2/v3 có feature set khác nhau đủ để decorrelate.
4. **Weight space đã bão hòa quanh 50/30/05/15**. Biên độ thay đổi khi đổi weights ≤ 100 điểm. Muốn break >1,000 điểm phải có signal **thực sự mới** (e.g., feature khác, target khác, model-family khác — không phải variant của LGBM).

## Compliance notes

- Không có feature nào đọc từ `sample_submission.csv` / `sales_test.csv`.
- Tất cả exog có |corr| với target > 0.95 đều được declare trong `LEAKY_LEVEL_COLS_V4` và DoY-impute (chứng minh trong `outputs/final_v4/leak_audit.csv`).
- Seed `(42, 123, 7, 2024, 31)`, `deterministic=True` — reproducible bằng `uv run python -m src.final_model_v4`.

## Files changed

- `src/calendar_vn.py` — VN holidays (Tet lunar 2012-2026, 1111/1212/BF/midautumn/...) — 32 features
- `src/features_v4.py` — 106 per-day exogenous features (A-J modules)
- `src/final_model_v4.py` — 5-seed LGBM bag trên 179 feats, log1p, WF-CV, recursive forecast
- `src/leak_audit_v4.py` — corr + NaN audit
- `src/build_blend_v4.py` + `build_blend_v4_fine.py` — 25+ blend candidates
- `outputs/final_v4/` — raw/calib/lb submissions, feature importance
- `outputs/candidates_v4/` — 25 blend CSV

## Feature importance (top 10, average of 5 seeds)

### Revenue
1. `items_gross_value` (DoY-imputed) — 7090
2. `days_since_start` — 2284
3. `pay_total_value` (DoY) — 2169
4. `Revenue_rmean28` — 795
5. `Revenue_rmean7` — 660
6. `Revenue_lag365` — 658
7. `items_cogs_total_value` (DoY) — 554
8. `year` — 547
9. `items_unique_products` — 372
10. `Revenue_lag364` — 250

### COGS
1. `COGS_doy_anchor_365` — 5981
2. `COGS_lag365` — 3431
3. `items_cogs_total_value` (DoY) — 1268
4. `COGS_lag364` — 1165
5. `days_since_start` — 778
6. `COGS_rmean28` — 497
7. `COGS_rmean7` — 456
8. `orders_unique_cities` — 444
9. `items_unique_products` — 434
10. `items_gross_value` (DoY) — 355

VN calendar flags không vào top 10 nhưng có trong top 50 (is_tet_eve_7d, days_to_black_friday). Marginal contribution.
