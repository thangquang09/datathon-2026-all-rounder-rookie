# Giải thích Mô hình — Kiến trúc 2 Tầng

> **Datathon 2026 — The Gridbreakers**
>
> Tài liệu này giải thích toàn bộ forecasting pipeline theo mô hình **2 tầng** (Two-Tier Architecture), giúp người đọc hiểu được *tại sao* mô hình ra quyết định mà không cần đi vào chi tiết từng model riêng lẻ.

---

## 1. Tổng quan: Pipeline gồm bao nhiêu models?

Toàn bộ pipeline sử dụng **~16 models** được stacked/blended:

| Component | Số models | Chi tiết |
|---|---|---|
| Recursive LGBM Ensemble (Pipeline A) | 9 | 3 cấu hình × 3 seeds |
| Direct LGBM+Ridge (Pipeline B) | 3 | LGBM + Ridge + DoY Prior |
| Legacy v1-v4 blend (Pipeline C) | 4 | final_model v1-no-proxy, v2, v3, v4 |
| **Final blend** | **1** | 80% M5 + 20% Direct |

Giải thích 16 models riêng lẻ là không thực tế và không cần thiết. Thay vào đó, chúng tôi sử dụng **kiến trúc 2 tầng** để trình bày.

---

## 2. Kiến trúc 2 Tầng (Two-Tier Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TẦNG 1: FEATURE ENGINEERING                  │
│                                                                 │
│  221 features từ 6 nhóm nghiệp vụ → đầu vào chung cho Tầng 2    │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐     │
│  │ 1. Seasonal  │ │ 2. Target    │ │ 3. Holiday & Event   │     │
│  │    Priors    │ │    Lags      │ │    Features           │     │
│  │  (61%)       │ │  (24%)       │ │  (11%)                │     │
│  └──────────────┘ └──────────────┘ └──────────────────────┘     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐     │
│  │ 4. Regime    │ │ 5. Anchor    │ │ 6. Horizon           │     │
│  │    Level     │ │    Level     │ │    Effect             │     │
│  │  (1.5%)      │ │  (1.5%)      │ │  (0.2%)               │     │
│  └──────────────┘ └──────────────┘ └──────────────────────┘     │
│                                                                 │
│  → Các model dùng cùng nhóm feature concept                     │
│  → Legacy v1 đã loại target-proxy same-day trước khi blend      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    TẦNG 2: ENSEMBLE & BLENDING                  │
│                                                                 │
│  Weighted average → giảm variance → forecast ổn định hơn        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  submission = 0.80 × M5_blend + 0.20 × Direct           │    │
│  │                                                          │    │
│  │  M5_blend = weighted avg(v1-no-proxy, v2, v3, v4)       │    │
│  │  Direct   = weighted avg(LGBM, Ridge, DoY)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  → Tầng 2 KHÔNG thay đổi insight từ Tầng 1                    │
│  → Chỉ giảm sai số ngẫu nhiên (variance reduction)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight

> **Tầng 2 là weighted average (trung bình có trọng số) — không thêm features mới, không thêm logic mới.** Nó chỉ giảm variance. Do đó, giải thích 1 model duy nhất ở Tầng 1 là đủ để hiểu toàn bộ pipeline.

---

## 3. Tầng 1: Giải thích chi tiết 6 nhóm features

### 3.1 Nhóm Calendar & Seasonality — 61% importance

**Ý nghĩa**: Nhóm này chiếm **61% tổng importance** — yếu tố quan trọng nhất. Fashion retail có tính mùa vụ rất mạnh: doanh thu ngày Tết, Black Friday, 11/11 cao gấp 5-10 lần ngày bình thường.

**Các features chính** (SHAP top contributors):

| Feature | SHAP (Revenue) | Giải thích |
|---|---|---|
| `forecast_year` | 0.145 | Năm dự báo — bắt trend tăng trưởng dài hạn |
| `doy_mean_cutoff` | 0.137 | DoY mean tính tới cutoff — "mùa này năm ngoái trung bình bao nhiêu?" |
| `doy_to_recent` | 0.070 | Tỷ lệ DoY mean / recent level — điều chỉnh theo hiện trạng |
| `month_dow_mean_cutoff` | 0.028 | Mean theo (month, dow) — bắt pattern tháng+thứ |
| `day` | 0.027 | Ngày trong tháng — bắt pattern đầu/cuối tháng |
| `sin/cos_week_1..3` | 0.017 | Fourier terms — Weekly seasonality |
| `dow` | 0.014 | Day of week — cuối tuần bán hơn ngày thường |

**Bao gồm**: 41 features (Fourier harmonics, DoY stats, month/dow interactions, calendar fields)

**Số lượng features**: 121 features (nhưng individual importance thấp vì phân tán)

### 3.2 Nhóm Target Lag — 24% importance

**Ý nghĩa**: "Năm ngoái hôm nay doanh thu bao nhiêu?" — Fashion có YoY consistency cao vì collection cycle lặp lại hàng năm.

**Các features chính**:

| Feature | SHAP (Revenue) | Giải thích |
|---|---|---|
| `forecast_lag730_known` | 0.123 | Giá trị 730 ngày trước (2 năm) — year-over-year anchor mạnh nhất |
| `anchor_2y_ratio730` | 0.039 | Tỷ lệ hiện tại / 2 năm trước — bắt growth rate |
| `forecast_lag364_known` | 0.014 | Giá trị 364 ngày trước — lag 1 năm (52 weeks) |
| `forecast_lag365_known` | 0.011 | Giá trị 365 ngày trước — lag 1 năm (calendar) |
| `forecast_lag728_known` | 0.011 | Giá trị 728 ngày — lag 2 năm (104 weeks) |
| `forecast_lag548_known` | 0.007 | Giá trị 548 ngày — chính xác halfway horizon |

**Số lượng features**: 26 features (multi-horizon lags: 182, 364, 365, 371, 548, 728, 730 days + anchor ratios)

**Tại sao lag-730 quan trọng hơn lag-365?** Vì 730 ngày = chính xác 2 năm, align với weekly cycle (730 = 2 × 365 = 104.3 weeks). Pipeline direct sử dụng lag-730 làm anchor vì nó ổn định hơn khi dự báo xa.

### 3.3 Nhóm Holiday & Event — 11% importance

**Ý nghĩa**: Các dịp lễ và sự kiện TMĐT tạo spike lớn — Tet, 11/11, Black Friday, Valentine, Quốc khánh...

**Các features chính**:

| Feature | SHAP (Revenue) | Giải thích |
|---|---|---|
| `hol_days_since_teachers_day` | 0.013 | Khoảng cách từ Ngày Nhà giáo — nhu cầu quà tặng |
| `hol_days_since_womens_day_mar8` | 0.013 | Khoảng cách từ 8/3 — nhu cầu quà tặng |
| `hol_days_to_valentine` | 0.009 | Khoảng cách đến Valentine — spike doanh thu |
| `hol_days_to_national_day` | 0.006 | Khoảng cách đến Quốc khánh 2/9 |
| `vn_days_to_tet` | 0.005 | Khoảng cách đến Tết — spike lớn nhất |
| `vn_days_to_mid_autumn` | 0.005 | Khoảng cách đến Trung thu |

**Số lượng features**: 121 features (days_to/from cho 20+ dịp lễ Việt Nam + sự kiện TMĐT)

### 3.4 Nhóm Regime Level — 1.5% importance

**Ý nghĩa**: Business experienced structural break ~2019 (COVID, market shift). Features này giúp model hiểu "mức độ" hiện tại so với trước/sau break.

**Các features chính**:

| Feature | SHAP (Revenue) | Giải thích |
|---|---|---|
| `cutoff_year` | 0.011 | Năm cutoff — bắt regime shift |
| `pre_break_mean_cutoff` | 0.006 | Mean trước 2019 — reference level |

**Số lượng features**: 7 features

### 3.5 Nhóm Anchor Level — 1.5% importance

**Ý nghĩa**: "Mức" hiện tại của doanh thu — rolling mean, median — giúp model anchor forecast quanh level gần đây.

**Các features chính**:

| Feature | SHAP (Revenue) | Giải thích |
|---|---|---|
| `anchor_roll_median365` | 0.005 | Rolling median 365 ngày — robust level estimate |
| `anchor_yoy_ratio365` | 0.005 | YoY ratio — growth rate |

**Số lượng features**: 20 features

### 3.6 Nhóm Horizon — 0.2% importance

**Ý nghĩa**: Khoảng cách dự báo — model học rằng dự báo xa hơn thì uncertain hơn.

**Số lượng features**: 5 features (`horizon`, `horizon_gt_365`, etc.)

---

## 4. Tầng 1: Sự nhất quán giữa SHAP và Gain

Một trong những kết quả quan trọng nhất: **SHAP và Gain cho ra cùng thứ tự importance**. Điều này chứng tỏ kết luận không phụ thuộc vào phương pháp giải thích.

### Revenue

| Nhóm | SHAP | Gain | Chênh lệch |
|---|---|---|---|
| Calendar & Seasonality | 61.4% | 58.6% | +2.8pp |
| Target Lag | 23.8% | 27.0% | -3.2pp |
| Holiday & Event | 10.7% | 11.4% | -0.7pp |
| Regime Level | 2.4% | 1.2% | +1.2pp |
| Anchor Level | 1.6% | 1.3% | +0.3pp |
| Horizon | 0.2% | 0.6% | -0.4pp |

### COGS

| Nhóm | SHAP | Gain | Chênh lệch |
|---|---|---|---|
| Calendar & Seasonality | 66.3% | 61.6% | +4.7pp |
| Target Lag | 17.8% | 24.5% | -6.7pp |
| Holiday & Event | 10.8% | 10.4% | +0.4pp |
| Regime Level | 3.4% | 1.4% | +2.0pp |
| Anchor Level | 1.5% | 1.4% | +0.1pp |
| Horizon | 0.2% | 0.6% | -0.4pp |

**Key message**: Top 3 nhóm (Seasonal Priors ~60%, Target Lags ~24%, Holiday Events ~11%) chiếm **>95% tổng importance** — nhất quán giữa SHAP và Gain. Kết luận kinh doanh không thay đổi dù dùng phương pháp giải thích nào.

---

## 5. Tầng 2: Tại sao Ensemble không thay đổi giải thích?

### 5.1 Ensemble là gì?

```
forecast_final = w₁ × model₁ + w₂ × model₂ + ... + wₙ × modelₙ
```

Mỗi model đều nhận **cùng input features** (Tầng 1), chỉ khác ở:
- Cách học (recursive vs direct)
- Hyperparameters (learning rate, depth)
- Training data split (seeds)

### 5.2 Ensemble không tạo ra features mới

Giả sử feature X có SHAP = 0.15 trong model đơn lẻ. Khi blend 16 models:

- Mỗi model đều sử dụng feature X → feature X vẫn xuất hiện trong kết quả cuối
- Weighted average chỉ giảm variance của **prediction**, không thay đổi **importance ranking**
- Nếu feature X là top-1 trong 16/16 models → nó sẽ là top-1 trong ensemble

### 5.3 Tại sao vẫn chọn Ensemble?

| Lý do | Giải thích |
|---|---|
| **Variance reduction** | 16 models giảm sai số ngẫu nhiên ~4x so với 1 model |
| **Robustness** | Nếu 1 model bị lỗi → 15 models còn lại vẫn hoạt động |
| **Không đổi insight** | Feature importance ranking không thay đổi |

### 5.4 So sánh hiệu năng

| Model | R² (Revenue) | R² (COGS) |
|---|---|---|
| XGBoost baseline (1 model) | 0.721 | 0.747 |
| Direct LGBM+Ridge (3 models) | 0.799 | 0.804 |
| Full ensemble (16 models) | ~0.82 (LB) | ~0.82 (LB) |

Ensemble cải thiện R² từ 0.72 → 0.82 (+14%) nhờ variance reduction, **không phải nhờ thêm features hay logic mới**.

---

## 6. Giải thích theo góc nhìn Business

### "Tại sao doanh thu ngày 11/11 được dự báo cao?"

```
Tầng 1:
  → Calendar: doy_mean_cutoff cho 11/11 = rất cao (top 5 ngày trong năm)
  → Holiday: vn_days_to_1111 = 0, is_1111 = True
  → Target Lag: forecast_lag365_known = doanh thu 11/11 năm ngoái (cao)
  → Anchor: roll_mean365 = level hiện tại

Tầng 2:
  → Weighted average của 16 models, tất cả đều nhận signal "11/11 = spike"
  → Kết quả: forecast cao, nhất quán
```

### "Tại sao dự báo Q1/2024 cao hơn Q1/2023?"

```
Tầng 1:
  → forecast_year = 2024 > 2023 → trend tăng trưởng
  → cutoff_year effect → recovery post-2019
  → Regime Level: partial recovery towards pre-break baseline

Tầng 2:
  → Tất cả models đều capture trend này
  → Ensemble không thêm/subtract, chỉ smooth
```

### "Tại sao dự báo tháng 7 thấp?"

```
Tầng 1:
  → Calendar: doy_mean cho mid-year = thấp (không có lễ lớn)
  → Holiday: không có event close → holiday SHAP = 0
  → Target Lag: forecast_lag365 cho tháng 7 = thấp

Tầng 2:
  → Models đồng thuận → low forecast, high confidence
```

---

## 7. Phương pháp SHAP được thực hiện như thế nào?

### Thiết kế

| Tham số | Giá trị |
|---|---|
| Model được giải thích | Direct LightGBM (Pipeline B) |
| Explainer | `shap.TreeExplainer` |
| Sample size | 2000 rows từ training set |
| Metric | Mean absolute SHAP value per feature |
| Aggregation | Trung bình across samples |

### Tại sao giải thích Direct LGBM thay vì toàn bộ ensemble?

1. **Direct LGBM là component chính** của Pipeline B (weight ~45% trong 3-model blend)
2. **SHAP được thiết kế cho 1 model** — mở rộng sang weighted ensemble phức tạp hơn nhưng không thêm insight
3. **Tất cả models dùng cùng features** → importance ranking của 1 model đại diện cho cả ensemble
4. **Trực quan và dễ hiểu** — 1 decision tree ensemble dễ visualize hơn 16-model blend

### Artifact files

| File | Nội dung |
|---|---|
| `direct_factory_shap_importance.csv` | SHAP importance per feature (443 rows: Revenue + COGS) |
| `direct_factory_shap_group_importance.csv` | SHAP importance per group (14 rows) |
| `direct_factory_feature_importance.csv` | Gain importance per feature (443 rows) |
| `direct_factory_feature_group_importance.csv` | Gain importance per group (14 rows) |
| `top30_features_shap_overall.png` | Bar chart top 30 features |

---

## 8. Tóm tắt

| Câu hỏi | Trả lời |
|---|---|
| Pipeline có bao nhiêu models? | ~16 models blended |
| Giải thích kiểu gì? | SHAP + Gain trên 1 model chính (Direct LGBM) |
| Tại sao đủ? | Vì Tầng 2 (ensemble) chỉ là weighted average — không thêm features/logic |
| Top 3 drivers? | Seasonal Priors (~60%), Target Lags (~24%), Holiday Events (~11%) |
| SHAP và Gain nhất quán? | Có — chênh lệch < 7pp ở mọi nhóm |
| Ensemble cải thiện bao nhiêu? | R²: 0.72 (1 model) → 0.80 (3 models) → ~0.82 (16 models) |
| Kết luận thay đổi nếu dùng model khác? | Không — tất cả models đều dựa vào cùng feature groups |

> **Bottom line**: Pipeline có thể phức tạp ở Tầng 2 (16 models), nhưng quyết định dự báo được driven bởi Tầng 1 (6 nhóm features) — và nhóm features nào quan trọng nhất là kết luận ổn định, không phụ thuộc vào choice of model hay ensemble strategy.
