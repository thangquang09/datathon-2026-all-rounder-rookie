# Datathon 2026 — The Gridbreakers

Repository này là workspace của team **All-Rounder Rookie** cho VinUni Datathon 2026. Mục tiêu của project là biến dữ liệu vận hành của một fashion e-commerce platform thành hai nhóm deliverables chính: phân tích kinh doanh có thể hành động được và mô hình dự báo `Revenue`/`COGS` theo ngày.

## 1. Giới Thiệu Contest

Datathon 2026 — The Gridbreakers được tổ chức bởi **VinTelligence — VinUniversity Data Science & AI Club**. Trong bối cảnh bài toán, team đóng vai trò data science team của một doanh nghiệp thương mại điện tử thời trang Việt Nam, cần phân tích dữ liệu bán hàng, khách hàng, sản phẩm, tồn kho, khuyến mãi, logistics và traffic để đề xuất chiến lược kinh doanh.

Cuộc thi gồm ba phần:

| Phần | Nội dung                                                   | Điểm |
| ----- | ----------------------------------------------------------- | -----: |
| 1     | Multiple Choice Questions                                   |     20 |
| 2     | Exploratory Data Analysis, visualization, business insights |     60 |
| 3     | Revenue/COGS forecasting trên Kaggle                       |     20 |

Repository này tập trung vào phần 2 và phần 3: EDA, customer segmentation, LTV/profit optimization, product/inventory/churn analysis, và forecasting pipeline.

## 2. Bài Toán Và Data

### Forecasting Target

Mô hình dự báo hai target theo ngày:

| Target      | Ý nghĩa                              |
| ----------- | -------------------------------------- |
| `Revenue` | Doanh thu ngày cần dự báo          |
| `COGS`    | Cost of Goods Sold ngày cần dự báo |

Forecast horizon là `2023-01-01` đến `2024-07-01`, tổng cộng 548 ngày. Submission cuối cùng nằm tại:

```text
data/processed/sales_forecast_submission/submission.csv
```

### Raw Data

Raw contest data nằm trực tiếp trong `data/`:

| Nhóm        | Files                                                                                                      |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| Sales target | `sales.csv`, `sample_submission.csv`                                                                   |
| Transactions | `orders.csv`, `order_items.csv`, `payments.csv`, `shipments.csv`, `returns.csv`, `reviews.csv` |
| Master data  | `customers.csv`, `products.csv`, `promotions.csv`, `geography.csv`                                 |
| Operations   | `inventory.csv`, `web_traffic.csv`                                                                     |

Một số processed outputs phục vụ EDA/report hiện nằm trong `data/truong.le/`, `data/thang.quang/` và `data/processed/`. Các output quan trọng nhất nên được tái tạo từ notebooks/scripts thay vì chỉnh tay.

## 3. Project Tree Và Vai Trò

Team thống nhất chỉ dùng bốn core folders: `data`, `docs`, `notebooks`, `src`. Không tạo project con hoặc folder kết quả rải rác ở root.

```text
.
├── data/
│   ├── *.csv
│   ├── processed/
│   │   └── sales_forecast_submission/
│   ├── thang.quang/
│   └── truong.le/
│
├── docs/
│   ├── CUSTOMER_SEGMENTATION_REPORT.md
│   ├── MODEL_REPORT.md
│   ├── MODEL_DOCUMENTATION.md
│   ├── PROJECT_STRUCTURE_AND_PUSH_RULES.md
│   ├── agent_rules/
│   └── images/
│
├── notebooks/
│   ├── eda_segmentation.ipynb
│   ├── eda_product_insights.ipynb
│   ├── model_baseline_seasonal.ipynb
│   ├── model_lgbm_forecasting.ipynb
│   └── model_reproduce_best_kaggle_solution.ipynb
│
└── src/
    ├── eda/
    ├── features/
    ├── models/
    │   └── sales_forecasting/
    ├── utils/
    └── visualization/
```

### Vai Trò Các Folder

| Folder                 | Vai trò                                                              |
| ---------------------- | --------------------------------------------------------------------- |
| `data/`              | Dữ liệu thô, bảng đã xử lý, model artifacts, submission cuối |
| `docs/`              | Báo cáo, ghi chú audit, quy tắc, hình ảnh, tài liệu           |
| `notebooks/`         | EDA/modeling notebooks dùng để phân tích và trình bày         |
| `src/eda/`           | Hàm EDA/phân khúc khách hàng tái sử dụng được              |
| `src/features/`      | Feature engineering dùng lại cho modeling                           |
| `src/models/`        | Huấn luyện model, inference, blending, forecasting pipeline         |
| `src/utils/`         | Tiện ích dùng chung, ví dụ lịch Việt Nam                       |
| `src/visualization/` | Scripts tạo hình ảnh báo cáo và biểu đồ chẩn đoán model   |

Chi tiết rule đặt file và push code nằm ở:

```text
docs/PROJECT_STRUCTURE_AND_PUSH_RULES.md
```

## 4. Main EDA Story

Luận điểm chính của EDA là: **doanh nghiệp không chỉ cần tăng revenue, mà cần giảm profit leakage và phát triển profitable loyalty**.

### Customer Segmentation, LTV & Profit Optimization

Phân tích customer segmentation dùng `Customer Golden Table`, profit-based RFM segmentation và objective score. Segment cuối cùng gồm 6 nhóm: `Champions`, `Loyal`, `Potential`, `Need Attention`, `At Risk`, `Lost`.

Các insight chính:

| Insight                               |                                 Evidence |
| ------------------------------------- | ---------------------------------------: |
| Tổng customers có realized purchase |                                   89,988 |
| Tổng revenue                         |                               16.24B VND |
| Tổng profit                          |                                2.24B VND |
| Top 20% customers đóng góp profit  |                                   66.49% |
| Customers có profit âm              | 9,722 customers, tương đương 10.80% |
| Tổng profit âm                      |                               -61.7M VND |
| XGBoost early loyalty AUC             |                                    0.731 |

Hàm ý kinh doanh:

- `Champions` tạo phần lớn profit nên cần loyalty benefits, early access, personalized service, không nên lạm dụng deep discount.
- `Potential` cần nurture để tăng frequency nhưng phải có margin guardrail.
- `At Risk` và `Lost` nên dùng low-cost automation hoặc win-back có điều kiện, tránh đốt ngân sách vào nhóm có LTV thấp.
- Promotion cần được quản trị theo `incremental profit`, không chỉ theo conversion rate.

Report chính:

```text
docs/CUSTOMER_SEGMENTATION_REPORT.md
notebooks/eda_segmentation.ipynb
```

### Product, Inventory, UX, Churn Analysis

Các phân tích bổ trợ tập trung vào product mix, category profit, return/refund reason, inventory pressure, co-purchase behavior và UX friction. Một kết luận xuyên suốt là product/category performance không nên đọc bằng volume đơn thuần; cần kết hợp margin, return/refund, promotion dependency và stock/inventory signal.

Tài liệu liên quan:

```text
docs/PRODUCT_INSIGHTS_REPORT.md
docs/PRODUCT_INSIGHTS_AUDIT.md
docs/ECOMMERCE_PERFORMANCE_ANALYSIS_REPORT.md
docs/ECOMMERCE_EDA_INSIGHT_REPORT_PART_2.md
notebooks/eda_product_insights.ipynb
```

## 5. Forecasting Pipeline

Forecasting pipeline được thiết kế theo hướng leakage-safe. Tại mỗi ngày forecast, model chỉ được dùng thông tin đã biết trước ngày đó hoặc deterministic calendar features.

### Model Architecture

Pipeline cuối cùng kết hợp hai hướng dự báo:

```text
final = 80% M5-style seasonal/regime blend + 20% direct LightGBM regime model
```

| Component                 | Vai trò                                                       |
| ------------------------- | -------------------------------------------------------------- |
| Seasonal/recursive models | Giữ seasonal shape, yearly memory và level ổn định        |
| Direct LightGBM           | Học trực tiếp horizon dài 548 ngày, giảm recursive drift |
| DoY prior                 | Neo forecast vào seasonal pattern lịch sử                   |
| Ensemble/blending         | Giảm model-specific bias và variance                         |

### Kết Quả Validation

CV đệ quy:

| Target  | Model đệ quy tốt nhất |     MAE |    RMSE |    R2 |
| ------- | ------------------------- | ------: | ------: | ----: |
| Revenue | CV weighted ensemble      | 631,776 | 879,062 | 0.680 |
| COGS    | CV weighted ensemble      | 547,400 | 758,884 | 0.688 |

CV direct model:

| Target  | Model           | MAE trung bình | MAE độ lệch chuẩn | R2 trung bình |
| ------- | --------------- | --------------: | --------------------: | -------------: |
| Revenue | Direct LightGBM |         529,738 |                93,477 |          0.764 |
| COGS    | Direct LightGBM |         452,324 |                81,482 |          0.785 |

So sánh với baseline:

| Target  | Model baseline tốt nhất |     MAE |    RMSE |    R2 |
| ------- | ------------------------- | ------: | ------: | ----: |
| Revenue | XGBoost                   | 576,792 | 821,306 | 0.721 |
| COGS    | XGBoost                   | 487,975 | 683,595 | 0.747 |

Điểm public Kaggle mới nhất đã ghi nhận:

```text
730,067.90380
```

Tài liệu model chính:

```text
docs/MODEL_REPORT.md
docs/MODEL_DOCUMENTATION.md
docs/MODEL_EXPLAINABILITY.md
docs/CV_DATA_SPLIT.md
notebooks/model_reproduce_best_kaggle_solution.ipynb
```

## 6. Tái Tạo Kết Quả

### Môi Trường

Project dùng `uv` và Python `>=3.13`.

```bash
uv sync
```

### Tái Tạo Submission Cuối Từ Artifacts Có Sẵn

```bash
uv run python -m src.models.sales_forecasting.scripts.reproduce_submission --overwrite
```

Lệnh này xác thực và sao chép:

```text
data/processed/sales_forecast_submission/artifacts/final_candidates/submission_m5_lgb_direct_blend_80_20.csv
```

vào:

```text
data/processed/sales_forecast_submission/submission.csv
```

### Chạy Toàn Bộ Forecasting Package

Huấn luyện đầy đủ nặng hơn. Chỉ dùng khi cần tái tạo lại artifacts.

```bash
uv run python -m src.models.sales_forecasting.train_save_infer_blend --skip-visuals
```

Tạo hình ảnh trực quan cho model:

```bash
uv run python -m src.visualization.sales_forecast_feature_importance
uv run python -m src.visualization.sales_forecast_pipeline_flowchart
```

### Docker (Khuyên Dùng Cho Giám Khảo)

Build Docker image và chạy toàn bộ pipeline end-to-end (~18 phút trên CPU):

```bash
docker build -t datathon2026-forecast .
docker run --rm -v $(pwd)/output:/app/data/processed/sales_forecast_submission datathon2026-forecast
```

Kết quả nằm tại `output/submission.csv`. Pipeline train 7 bước tuần tự: recursive LGBM + seasonal/DoY ensemble → v4 regime → legacy v1–v4 blend → M5-style blend → direct LightGBM/Ridge factory → inference → final blend 80/20 → `submission.csv`.

## 7. Các Output Quan Trọng

| Output                             | Đường dẫn                                                                             |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| Submission cuối                   | `data/processed/sales_forecast_submission/submission.csv`                               |
| Forecast artifacts                 | `data/processed/sales_forecast_submission/artifacts/`                                   |
| SHAP/feature importance            | `data/processed/sales_forecast_submission/artifacts/direct_factory_shap_importance.csv` |
| Customer golden table              | `data/truong.le/customer_golden_table.csv`                                              |
| Bảng khách hàng có phân khúc | `data/truong.le/customer_golden_table_with_segments.csv`                                |
| Hình ảnh báo cáo               | `docs/images/`                                                                          |

## 8. Thành Viên

Team **All-Rounder Rookie**:

| Thành viên          | Vai trò chính                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Lê Phú Trường     | Trưởng nhóm, phân khúc khách hàng, tối ưu LTV/profit, báo cáo, cấu trúc codebase |
| Lý Quang Thắng      | Dự báo Revenue/COGS, feature engineering, model ensemble, Kaggle submission                 |
| Trần Lê Hữu Vinh   | Hỗ trợ EDA/phân tích sản phẩm và kinh doanh, phát triển model                        |
| Lê Đặng Gia Khánh | Hỗ trợ EDA/modeling/báo cáo                                                               |

## 9. Liên Kết Nhanh

| Nhu cầu                               | Bắt đầu từ đây                                     |
| -------------------------------------- | -------------------------------------------------------- |
| Hiểu quy tắc project                 | `docs/PROJECT_STRUCTURE_AND_PUSH_RULES.md`             |
| Đọc báo cáo phân khúc            | `docs/CUSTOMER_SEGMENTATION_REPORT.md`                 |
| Đọc báo cáo model                  | `docs/MODEL_REPORT.md`                                 |
| Tái tạo submission tốt nhất        | `notebooks/model_reproduce_best_kaggle_solution.ipynb` |
| Làm việc với EDA phân khúc        | `notebooks/eda_segmentation.ipynb`                     |
| Làm việc với phân tích sản phẩm | `notebooks/eda_product_insights.ipynb`                 |
| Thí nghiệm model                     | `notebooks/model_lgbm_forecasting.ipynb`               |
| Tái tạo submission bằng Docker  | `docker run --rm datathon2026-forecast`                 |
