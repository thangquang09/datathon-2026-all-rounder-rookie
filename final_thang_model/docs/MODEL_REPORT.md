# Báo cáo Model Dự báo Revenue/COGS — DATATHON 2026

**Ngày chạy pipeline:** 2026-05-01
**Notebook:** `notebooks/reproduce_best_kaggle_solution.ipynb`
**Submission:** `submission.csv`
**Kaggle public score đã submit:** **730,067.90380**

---

## 1. Mục tiêu và Tiêu chí

Theo `data/contest_rules.md`, phần modeling (Phần 3 — 20 điểm) gồm:

| Thành phần | Điểm tối đa |
|------------|-------------|
| Hiệu suất mô hình (leaderboard MAE, RMSE, R²) | 12 |
| Báo cáo kỹ thuật (pipeline, CV, SHAP/feature importance) | 8 |

Package này giải quyết cả hai: score leaderboard tốt bằng ensemble time-series, kèm đầy đủ feature engineering, validation, và explainability artifacts.

---

## 2. Submission Cuối

**File:** `final_thang_model/submission.csv`

**Kaggle submit:**

```
description: ablation drop v1 target proxy features
status: COMPLETE
publicScore: 730,067.90380
```

**Validation đã pass:**

- 548 dòng, cột `Date, Revenue, COGS`
- Từ `2023-01-01` đến `2024-07-01`
- Không missing, không infinite, Revenue/COGS đều dương
- Đúng thứ tự `sample_submission.csv`

---

## 3. Kiến trúc Model

```
final = 80% M5-style seasonal/regime blend + 20% direct LightGBM regime model
```

| Component | Mô tả | Vai trò |
|-----------|-------|---------|
| **M5-style blend** | Seasonal memory + yearly shape + regime recovery + calendar/holiday + blend theo năm gần nhất | Giữ seasonal shape ổn định, calibration level |
| **Direct LightGBM** | Dự báo trực tiếp từng horizon 548 ngày, dùng pseudo-cutoff lịch sử | Học residual theo horizon/calendar/known-lag |
| **Revenue & COGS** | Train riêng biệt | Tương quan cao nhưng biên lợi nhuận không cố định |

### M5-style Blend Composition

```
M5 blend = 50% v1-no-proxy + 30% v2 + 5% v3 + 15% v4
```

Trong đó v1–v4 là các iteration của model recursive với feature engineering khác nhau, tất cả được regime-recovery calibrated.
Legacy `v1` vẫn được giữ để tạo diversity, nhưng các target-proxy same-day đã
bị loại khỏi feature set: `items_gross_value`, `pay_total_value`,
`pay_mean_value`, `orders_count`, `items_total_qty` và các lag/rolling dẫn xuất
từ những cột này.

### Direct LightGBM Composition

```
Direct = 90% LightGBM + 10% DOY prior
(Ridge bị loại do R² âm trên tất cả folds)
```

### Final 80/20 Blend

```
submission.csv = 80% × M5_blend + 20% × Direct_LightGBM_regime_recovery
```

**Candidate file:** `model_thang/artifacts/advanced_experiments/submission_m5_lgb_direct_blend_80_20.csv`

---

## 4. Pipeline Reproduce

Notebook chạy 7 scripts theo thứ tự:

```
Step 1: forecast_pipeline.py          → base recursive/cv ensemble
Step 2: build_v4_regime_candidate.py  → v4 direct model
Step 3: build_legacy_blend_regime.py  → v1–v4 legacy components
Step 4: build_m5_style_blend.py       → M5 seasonal/regime blend
Step 5: explainable_forecast_factory.py → direct explainable + SHAP
Step 6: build_direct_lgb_candidates.py → final M5+LGB blends
Step 7: visualize_top_features.py     → charts
```

Tất cả script nằm trong `final_thang_model/`, chỉ đọc dữ liệu từ `../data/`.

---

## 5. Dữ liệu Sử dụng

| Nhóm | Files | Features sinh ra |
|------|-------|-----------------|
| **Sales target** | `sales.csv` | Revenue/COGS daily, lag, rolling, DOY prior |
| **Transaction** | `orders.csv`, `order_items.csv`, `products.csv` | Order count, quantity, category mix, discount |
| **Payment** | `payments.csv` | Payment value, installments, method mix |
| **Traffic** | `web_traffic.csv` | Sessions, visitors, page views, bounce rate |
| **Operations** | `returns.csv`, `reviews.csv`, `shipments.csv` | Refund, rating, shipping lead time |
| **Inventory** | `inventory.csv` | Stockout, overstock, sell-through, fill rate |
| **Promotions** | `promotions.csv` | Active promo intensity |
| **Customers** | `customers.csv` | Signup signals |
| **Holidays** | `docs/vietnam_calendar_events_deterministic_2012_2024.csv` | Lễ Việt Nam, retail events sinh từ dương lịch |

**Leakage guard:** `sample_submission.csv` chỉ đọc cột `Date`.
Không dùng target-proxy same-day trong legacy `v1`; các model v2/v3 drop proxy
và v4/direct dùng climatology/cutoff-safe features.

---

## 6. Feature Engineering

### 6.1 Calendar & Seasonality

- `year`, `month`, `quarter`, `week`, `day_of_week`, `day_of_year`, `weekend`, `month_start/end`, `payday_window`
- **Fourier:** `sin/cos_week_1..3`, `sin/cos_year_1..6` → chu kỳ tuần/năm mượt

### 6.2 Vietnam Holidays & Events

| Event | Feature | Ý nghĩa |
|-------|---------|----------|
| Tết Nguyên Đán | `vn_days_to_tet`, `vn_days_since_tet`, `vn_is_tet_window` | Spike trước Tết, drop trong Tết |
| Giỗ Tổ Hùng Vương | `vn_days_to_hung_kings` | Ngày lễ chính thức |
| 30/4 – 1/5 | `hol_days_to_liberation_day`, `hol_days_to_labour_day` | Kỳ nghỉ dài, demand shift |
| Quốc Khánh 2/9 | `hol_days_to_national_day` | Promotion window |
| Valentine 14/2 | `hol_days_to_valentine` | Gifting demand |
| Quốc tế Phụ nữ 8/3 | `hol_days_since_womens_day_mar8` | Fashion gifting |
| Ngày Nhà giáo 20/11 | `hol_days_since_teachers_day` | Late-year gifting |
| 11/11, Black Friday, 12/12 | event windows | Campaign peaks |
| Trung Thu | `vn_days_to_mid_autumn` | Seasonal demand |

### 6.3 Target Memory (Lag/Rolling)

- **Short lag (recursive):** lag 7, 14, 28
- **Yearly memory:** lag 364, 365, 371, 548, 728, 730
- **Rolling:** mean/std 7→365 ngày
- **Direct-horizon known lags:** chỉ dùng khi `lag_date <= cutoff`

### 6.4 Seasonal Priors

- `Revenue_doy_mean_cutoff` / `Revenue_doy_median_cutoff`: trung vị doanh thu cùng ngày trong năm (tính đến cutoff)
- `Revenue_month_dow_mean_cutoff`: trung bình theo (tháng, thứ)
- `Revenue_doy_to_recent`: tỷ lệ DOY prior / recent level → bắt seasonality shape thay vì level

### 6.5 Regime & Anchor

- `cutoff_year`, `Revenue_pre_break_mean_cutoff`: structural break 2019–2022 vs recovery 2022+
- `Revenue_anchor_2y_ratio730`: tỷ lệ so với 2 năm trước → bắt xu hướng tăng trưởng
- `Revenue_anchor_roll_median365`, `Revenue_anchor_yoy_ratio365`: anchor levels

### 6.6 Operational Lag Features

Order, traffic, payment, return, review, shipping, inventory, promotion, customer — tất cả ở dạng lag/climatology an toàn (không dùng future data).

---

## 7. Leakage Control

| Guard | Chi tiết |
|-------|----------|
| Không đọc target từ sample | Chỉ load cột `Date` từ `sample_submission.csv` |
| Recursive safe | Sau vài ngày đầu test, lag 7/14 lấy từ prediction đã sinh, không dùng actual future |
| Direct safe | Lag feature chỉ dùng khi `lag_date <= cutoff` |
| No future ops | Operational variables chỉ dùng lag/climatology từ history |
| Calendar deterministic | Holiday/calendar biết trước tại thời điểm forecast |
| Regime from train only | Yearly calibration chỉ dùng `sales.csv` 2014–2018 và 2022 recovery |

---

## 8. Kết quả Validation (chạy mới nhất)

### 8.1 Base Recursive/CV Ensemble (`forecast_pipeline.py`)

| Target | Model | MAE (trung bình CV) | RMSE (trung bình CV) | R² (trung bình CV) |
|--------|-------|--------------------:|---------------------:|-------------------:|
| Revenue | cv_weighted_ensemble | 631,775.84 | 879,062.00 | 0.6804 |
| Revenue | model (LightGBM) | 663,052.02 | 912,972.70 | 0.6550 |
| Revenue | seasonal | 828,617.70 | 1,218,866.00 | 0.3853 |
| Revenue | doy_prior | 682,579.04 | 933,400.80 | 0.6395 |
| COGS | cv_weighted_ensemble | 547,400.37 | 758,883.90 | 0.6876 |
| COGS | model (LightGBM) | 574,961.61 | 791,782.00 | 0.6600 |
| COGS | seasonal | 728,542.66 | 1,083,031.00 | 0.3624 |
| COGS | doy_prior | 594,279.63 | 818,336.70 | 0.6364 |

**CV weights:**

```
Revenue: 40% model + 30% seasonal + 30% doy
COGS:    35% model + 25% seasonal + 40% doy
```

### 8.2 Direct Explainable Factory (`explainable_forecast_factory.py`)

**Time-aware CV — 3 folds:**

| Target | Fold Cutoff | Model | MAE | RMSE | R² |
|--------|------------|-------|----:|-----:|---:|
| Revenue | 2020-06-30 | LightGBM | 528,581 | 800,948 | 0.7345 |
| Revenue | 2020-06-30 | Ridge | 2,479,777 | 2,876,758 | -2.4245 |
| Revenue | 2020-06-30 | DOY prior | 1,517,742 | 1,833,965 | -0.3918 |
| Revenue | 2020-12-31 | LightGBM | 601,202 | 861,855 | 0.7612 |
| Revenue | 2020-12-31 | Ridge | 2,849,044 | 3,274,444 | -2.4474 |
| Revenue | 2020-12-31 | DOY prior | 1,305,268 | 1,684,278 | 0.0879 |
| Revenue | 2021-07-01 | LightGBM | 441,182 | 664,638 | 0.8168 |
| Revenue | 2021-07-01 | Ridge | 677,453 | 961,442 | 0.6167 |
| Revenue | 2021-07-01 | DOY prior | 899,250 | 1,195,173 | 0.4077 |
| **Revenue** | **all_oof** | **weighted_direct** | **503,012** | **733,825** | **0.7994** |
| COGS | 2020-06-30 | LightGBM | 435,557 | 650,611 | 0.7661 |
| COGS | 2020-06-30 | Ridge | 1,994,446 | 2,313,639 | -1.9580 |
| COGS | 2020-06-30 | DOY prior | 1,290,190 | 1,557,013 | -0.3396 |
| COGS | 2020-12-31 | LightGBM | 535,902 | 761,583 | 0.7568 |
| COGS | 2020-12-31 | Ridge | 2,262,263 | 2,610,749 | -1.8584 |
| COGS | 2020-12-31 | DOY prior | 1,027,205 | 1,338,314 | 0.2489 |
| COGS | 2021-07-01 | LightGBM | 386,235 | 566,307 | 0.8285 |
| COGS | 2021-07-01 | Ridge | 544,513 | 744,370 | 0.7036 |
| COGS | 2021-07-01 | DOY prior | 756,748 | 1,009,937 | 0.4544 |
| **COGS** | **all_oof** | **weighted_direct** | **441,595** | **633,897** | **0.8038** |

**Direct component weights:**

```
Revenue: 90% LightGBM + 0% Ridge + 10% DOY prior
COGS:    90% LightGBM + 0% Ridge + 10% DOY prior
```

**Nhận xét:** Direct-horizon model (R² ≈ 0.80) tốt hơn rõ rệt recursive-only (R² ≈ 0.68), nhưng blend với M5-style seasonal/regime vẫn cho leaderboard tốt nhất.

---

## 9. Explainability

### 9.1 Top 30 Features (SHAP Overall)


| Rank | Feature | Group | Mean SHAP | Gain |
|-----:|---------|-------|----------:|-----:|
| 1 | `forecast_year` | calendar_seasonality | 0.2866 | 71,251 |
| 2 | `Revenue_doy_mean_cutoff` | calendar_seasonality | 0.1368 | 66,012 |
| 3 | `Revenue_forecast_lag730_known` | target_lag | 0.1228 | 79,579 |
| 4 | `COGS_doy_mean_cutoff` | calendar_seasonality | 0.1227 | 65,017 |
| 5 | `COGS_doy_to_recent` | calendar_seasonality | 0.1199 | 64,835 |
| 6 | `Revenue_doy_to_recent` | calendar_seasonality | 0.0703 | 37,057 |
| 7 | `day` | calendar_seasonality | 0.0664 | 41,822 |
| 8 | `COGS_forecast_lag730_known` | target_lag | 0.0458 | 45,900 |
| 9 | `Revenue_anchor_2y_ratio730` | target_lag | 0.0394 | 9,402 |
| 10 | `cutoff_year` | regime_level | 0.0375 | 11,315 |
| 11 | `COGS_anchor_2y_ratio730` | target_lag | 0.0374 | 10,951 |
| 12 | `cos_week_1` | calendar_seasonality | 0.0354 | 50,296 |
| 13 | `sin_week_1` | calendar_seasonality | 0.0334 | 55,694 |
| 14 | `Revenue_month_dow_mean_cutoff` | calendar_seasonality | 0.0279 | 19,958 |
| 15 | `COGS_forecast_lag365_known` | target_lag | 0.0277 | 30,495 |
| 16 | `COGS_month_dow_mean_cutoff` | calendar_seasonality | 0.0247 | 20,017 |
| 17 | `dow` | calendar_seasonality | 0.0247 | 9,214 |
| 18 | `Revenue_doy_median_cutoff` | calendar_seasonality | 0.0242 | 21,284 |
| 19 | `hol_days_to_valentine` | holiday_event | 0.0234 | 5,112 |
| 20 | `hol_days_since_womens_day_mar8` | holiday_event | 0.0220 | 6,927 |
| 21 | `COGS_forecast_lag364_known` | target_lag | 0.0200 | 31,763 |
| 22 | `COGS_doy_median_cutoff` | calendar_seasonality | 0.0197 | 21,283 |
| 23 | `hol_days_since_teachers_day` | holiday_event | 0.0185 | 5,761 |
| 24 | `Revenue_forecast_lag364_known` | target_lag | 0.0139 | 28,288 |
| 25 | `sin_week_2` | calendar_seasonality | 0.0139 | 47,301 |
| 26 | `hol_days_to_national_day` | holiday_event | 0.0131 | 8,221 |
| 27 | `cos_year_6` | calendar_seasonality | 0.0129 | 26,554 |
| 28 | `sin_year_6` | calendar_seasonality | 0.0127 | 37,449 |
| 29 | `Revenue_forecast_lag365_known` | target_lag | 0.0115 | 24,345 |
| 30 | `Revenue_forecast_lag728_known` | target_lag | 0.0108 | 34,401 |


### 9.2 Feature Group Importance

**Theo LightGBM Gain:**

| Target | Group | Gain | % Total |
|--------|-------|-----:|--------:|
| Revenue | calendar_seasonality | 578,926 | 57.1% |
| Revenue | target_lag | 266,505 | 26.3% |
| Revenue | holiday_event | 112,893 | 11.1% |
| Revenue | anchor_level | 12,993 | 1.3% |
| Revenue | regime_level | 11,461 | 1.1% |
| Revenue | horizon | 5,562 | 0.5% |
| COGS | calendar_seasonality | 596,431 | 58.7% |
| COGS | target_lag | 237,498 | 23.4% |
| COGS | holiday_event | 100,588 | 9.9% |
| COGS | regime_level | 13,944 | 1.4% |
| COGS | anchor_level | 13,159 | 1.3% |
| COGS | horizon | 5,762 | 0.6% |

**Theo SHAP Mean Abs Value:**

| Target | Group | Mean SHAP | % Total |
|--------|-------|----------:|--------:|
| Revenue | calendar_seasonality | 0.5776 | 57.5% |
| Revenue | target_lag | 0.2241 | 22.3% |
| Revenue | holiday_event | 0.1009 | 10.1% |
| Revenue | regime_level | 0.0222 | 2.2% |
| Revenue | anchor_level | 0.0146 | 1.5% |
| COGS | calendar_seasonality | 0.6093 | 60.5% |
| COGS | target_lag | 0.1638 | 16.3% |
| COGS | holiday_event | 0.0992 | 9.9% |
| COGS | regime_level | 0.0315 | 3.1% |
| COGS | anchor_level | 0.0134 | 1.3% |

### 9.3 Diễn giải Kinh doanh

**1. Calendar & Seasonality (~58% importance)**

Demand thời trang có chu kỳ năm rất mạnh. `forecast_year` là feature quan trọng nhất — cho thấy xu hướng tăng trưởng theo năm. `doy_mean_cutoff` và `month_dow_mean_cutoff` lưu giữ "shape" bán hàng theo ngày trong năm và (tháng, thứ). Fourier features (`sin/cos_week`, `sin/cos_year`) giúp model nội suy mượt giữa các ngày.

**→ Bài học:** Dữ liệu bán lẻ thời trang Việt Nam có seasonality mạnh theo lịch. Model dự báo tốt nhất khi nắm chắc "hình dáng" năm, không chỉ dựa vào xu hướng gần nhất.

**2. Target Lag & Memory (~24% importance)**

`forecast_lag730_known` (2 năm trước) quan trọng hơn lag 365/364 (1 năm) — data có chu kỳ 2 năm rõ. `anchor_2y_ratio730` cho phép model tự điều chỉnh theo tỷ lệ tăng trưởng YoY. `doy_to_recent` tỷ lệ giữa prior DOY và recent level, giúp tách shape vs level.

**→ Bài học:** Revenue/COGS có tính "memory" mạnh — doanh thu cùng ngày năm trước là dự báo baseline tốt nhất, nhưng cần điều chỉnh theo trend.

**3. Holiday & Events (~10% importance)**

Valentine, Quốc tế Phụ nữ 8/3, Ngày Nhà giáo 20/11, Quốc Khánh 2/9, Tết, Trung Thu — tất cả đều có impact measurable. `days_to_event` và `days_since_event` bắt được demand uplift/drop quanh các dịp này.

**→ Bài học:** Lễ và campaign ngày là driver quan trọng. Đây là cơ hội để team marketing/commercial plan trước inventory và promotion.

**4. Regime Level (~2% importance)**

Dữ liệu có structural break 2019–2022 (COVID era) và recovery 2022+. Model tách riêng pre-break mean và cutoff year để không bị kéo xuống bởi giai đoạn bất thường.

**→ Bài học:** Yearly calibration từ train data cho phép model dự báo recovery 2023–2024, thay vì lặp lại level thấp 2019–2022.

**5. Horizon & Anchor (~1% importance)**

Direct model học behavior khác nhau cho short-term vs long-term forecast, giảm recursive drift. Anchor level từ rolling median 365 ngày giữ forecast stable.

### 9.4 Visualizations

Các chart đã sinh:

| File | Nội dung |
|------|----------|
| `feature_importance/top30_features_shap_overall.png` | Top 30 features theo SHAP (gộp Revenue + COGS) |
| `feature_importance/top30_features_shap_by_target.png` | Top 30 features SHAP tách riêng Revenue vs COGS |
| `feature_importance/top30_features_gain_overall.png` | Top 30 features theo LightGBM Gain |
| `feature_importance/top30_feature_shortlist.csv` | Top 30 features + group + SHAP + Gain |

### 9.5 Explainability Artifacts

| File | Nội dung |
|------|----------|
| `direct_factory_cv_metrics.csv` | Chi tiết CV metrics từng fold, từng model |
| `direct_factory_feature_importance.csv` | Full feature importance by gain |
| `direct_factory_shap_importance.csv` | Full SHAP importance từng feature (443 features) |
| `direct_factory_shap_group_importance.csv` | SHAP importance theo group |
| `direct_factory_feature_group_importance.csv` | Gain importance theo group |
| `direct_factory_component_debug.csv` | Debug info từng component |
| `explainable_forecast_factory_report.md` | Báo cáo kỹ thuật factory |
| `direct_factory_audit.json` | Audit metadata |

---

## 10. Vì sao Ensemble vẫn Giải thích được

| Component | Vai trò | Giải thích được? |
|-----------|---------|:---:|
| M5-style blend | Giữ seasonal shape + regime calibration | Yes — deterministic rules |
| Direct LightGBM | Học residual theo horizon/calendar/lag | Yes — SHAP + gain |
| Calendar/holiday | Deterministic | Yes — auditable |
| Regime recovery | Train-only calibration | Yes — no leakage |

**Tóm lại:** Model cuối không chỉ tối ưu leaderboard mà còn có câu chuyện kinh doanh rõ ràng — dữ liệu daily commerce có seasonality mạnh, bị structural break/recovery, và hai target cần học riêng nhưng cùng chia sẻ calendar, holiday, yearly memory và operational priors.

---

## 11. Cấu trúc Folder

```
final_thang_model/
├── train_save_infer_blend.py               ← Script chính train/save/infer/blend
├── submission.csv                          ← Final submission (80/20 blend)
├── README.md
├── notebooks/
│   └── reproduce_best_kaggle_solution.ipynb
├── scripts/
│   ├── reproduce_submission.py
│   ├── run_baselines.py
│   └── generate_flowchart.py
├── docs/
│   ├── MODEL_REPORT.md                     ← Báo cáo này
│   ├── MODEL_DOCUMENTATION.md
│   ├── MODEL_EXPLAINABILITY.md
│   ├── CV_DATA_SPLIT.md
│   ├── vietnam_calendar_events_deterministic_2012_2024.csv
│   ├── VIETNAM_HOLIDAY_FEATURE_AUDIT.md
│   ├── assets/
│   │   └── PIPELINE_FLOWCHART.png
│   └── tables/
│       ├── baseline_results.csv
│       ├── pipeline_results.csv
│       └── full_feature_importance.csv
├── model_thang/
│   ├── forecast_pipeline.py
│   ├── build_v4_regime_candidate.py
│   ├── build_legacy_blend_regime.py
│   ├── build_m5_style_blend.py
│   ├── explainable_forecast_factory.py
│   ├── build_direct_lgb_candidates.py
│   ├── visualize_top_features.py
│   └── artifacts/
│       ├── cv_metrics.csv
│       ├── cv_weights.json
│       ├── direct_factory_cv_metrics.csv
│       ├── direct_factory_shap_importance.csv
│       ├── direct_factory_feature_importance.csv
│       ├── direct_factory_feature_group_importance.csv
│       ├── direct_factory_shap_group_importance.csv
│       ├── direct_factory_component_debug.csv
│       ├── direct_factory_audit.json
│       ├── explainable_forecast_factory_report.md
│       ├── feature_importance_gain.csv
│       ├── feature_importance/
│       │   ├── top30_features_shap_overall.png
│       │   ├── top30_features_shap_by_target.png
│       │   ├── top30_features_gain_overall.png
│       │   └── top30_feature_shortlist.csv
│       ├── advanced_experiments/
│       │   ├── submission_m5_lgb_direct_blend_80_20.csv  ← final candidate
│       │   └── ... (other blend ratios)
│       └── ... (component CSVs, regime files)
└── src/
    ├── calendar_vn.py
    ├── features_v4.py
    ├── final_model.py
    ├── final_model_v2.py
    ├── final_model_v3.py
    └── final_model_v4.py
```

**Chỉ cần thêm `data/` folder là chạy được.**
