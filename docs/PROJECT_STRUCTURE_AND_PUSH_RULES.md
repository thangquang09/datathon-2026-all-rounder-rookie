# Project Structure and Push Rules

## Mục tiêu

Repository này là nơi làm việc chung cho datathon, mọi file được push lên phải giúp team hiểu nhanh: file này thuộc phần nào, dùng để làm gì, ai có thể tái sử dụng, và có nên xuất hiện trong final deliverable hay không.

Từ thời điểm áp dụng guideline này, team chỉ sử dụng bốn core folders chính cho sản phẩm làm việc:

```text
.
├── data/
├── docs/
├── notebooks/
└── src/
```

Không tạo thêm folder ngẫu nhiên ở root như `report/`, `results_*`, `final_*`, `temp_*`, `new/`, `draft/`, `abc/`, `test/`, `v1/`, `v2/`, hoặc folder đặt theo tên người nếu không có lý do rõ ràng và chưa được thống nhất. Nếu cần thử nghiệm cá nhân, dùng branch riêng hoặc local-only files đã được ignore, không đẩy lên `main`.

## Cấu trúc chuẩn

Project tree chuẩn nên đi theo hướng sau:

```text
.
├── data/
│   ├── raw/                         # Optional: raw input nếu cần tách khỏi root data
│   ├── interim/                     # Optional: intermediate data có thể tái tạo
│   ├── processed/                   # Optional: final processed data cho notebook/model
│   ├── external/                    # Optional: external assets được phép dùng, nếu có
│   ├── truong.le/                   # Existing personal output, cần migrate dần nếu dùng chung
│   └── README.md                    # Data dictionary và rule sử dụng data
│
├── docs/
│   ├── CUSTOMER_SEGMENTATION_REPORT.md
│   ├── PROJECT_STRUCTURE_AND_PUSH_RULES.md
│   ├── assets/
│   │   └── customer_segmentation_ltv/
│   └── reports/                     # Optional: final/submission reports nếu cần tách
│
├── notebooks/
│   ├── eda_segmentation.ipynb
│   ├── eda_product_insights.ipynb
│   ├── model_sales_forecasting.ipynb
│   └── README.md                    # Optional: notebook index
│
└── src/
    ├── __init__.py
    ├── eda/
    │   ├── __init__.py
    │   ├── customer_distribution_map.py
    │   ├── rfm_objective_segmentation.py
    │   └── segment_behavior_profiles.py
    ├── features/
    │   └── sales_features.py
    ├── models/
    │   └── sales_forecasting.py
    ├── visualization/
    │   └── report_figures.py
    └── utils/
        └── calendar_vn.py
```

Đây là cấu trúc định hướng. Không bắt buộc tạo tất cả folder ngay lập tức. Chỉ tạo folder khi có ít nhất một file thật sự cần đặt vào đó.

## Quy định cho `data/`

`data/` chỉ chứa dữ liệu đầu vào, dữ liệu trung gian có thể tái tạo, hoặc output dạng bảng được nhiều người dùng lại.

Tên file trong `data/` phải nói rõ nội dung và stage xử lý. Ví dụ tốt:

```text
data/processed/customer_golden_table.csv
data/processed/customer_golden_table_with_segments.csv
data/interim/order_item_facts.csv
data/processed/loyalty_model_coefficients.csv
```

Ví dụ không được chấp nhận:

```text
data/final.csv
data/result.csv
data/new_data.csv
data/test.csv
data/abc.csv
data/output_1.csv
```

Không push data lớn, data duplicate, hoặc file sinh ra tạm thời nếu có thể tạo lại bằng notebook/script. Nếu bắt buộc push processed data, phải có code tạo ra nó hoặc ghi rõ nguồn trong docs/notebook.

Không sửa raw data gốc. Nếu cần clean data, tạo file mới trong `data/interim/` hoặc `data/processed/`, không overwrite file gốc.

## Quy định cho `docs/`

Mọi Markdown, report draft, analysis write-up, rubric note, decision log và documentation đều phải nằm trong `docs/`.

Tên file Markdown phải có ý nghĩa và phản ánh nội dung. Ví dụ tốt:

```text
docs/Customer Segmentation, LTV & Profit Optimization.md
docs/PROJECT_STRUCTURE_AND_PUSH_RULES.md
docs/EDA_SCORING_RUBRIC_NOTES.md
docs/SALES_FORECASTING_MODEL_NOTES.md
docs/assets/customer_segmentation_ltv/01_profit_lorenz_curve.png
```

Ví dụ không được chấp nhận:

```text
report.md
summary.md
final.md
notes.md
results.md
bao_cao.md
docs/report1.md
docs/final_final.md
docs/new.md
```

Nếu Markdown dùng hình ảnh, hình ảnh phải nằm trong `docs/assets/<topic_name>/`. Không để ảnh rải ở root, trong notebook folder, hoặc trong folder `results_*`.

Mọi report final hoặc near-final phải có source rõ ràng. Nếu file PDF/TEX được build từ Markdown hoặc script, phải chỉ ra script/notebook tạo ra nó.

## Quy định cho `notebooks/`

Mọi notebook phải nằm trong `notebooks/`. Không tạo notebook ở root, trong `src/`, trong `docs/`, hoặc trong folder kết quả cá nhân.

Notebook phải có prefix theo intent:

```text
notebooks/eda_*.ipynb       # Exploratory Data Analysis
notebooks/model_*.ipynb     # Modeling experiments
notebooks/feat_*.ipynb      # Feature engineering exploration
notebooks/audit_*.ipynb     # Data/model validation, leakage check, metric audit
notebooks/report_*.ipynb    # Notebook tạo figure/table cho report
```

Ví dụ tốt:

```text
notebooks/eda_segmentation.ipynb
notebooks/eda_product_insights.ipynb
notebooks/model_sales_forecasting.ipynb
notebooks/audit_sales_leakage.ipynb
notebooks/report_customer_ltv_figures.ipynb
```

Ví dụ không được chấp nhận:

```text
notebooks/baseline.ipynb
notebooks/test.ipynb
notebooks/Untitled.ipynb
notebooks/new.ipynb
notebooks/final.ipynb
notebooks/eda_v1.ipynb
notebooks/eda_v2.ipynb
notebooks/eda_final_final.ipynb
```

Không tạo nhiều notebooks có intent trùng lặp. Nếu đã có `eda_segmentation.ipynb`, không tạo thêm `eda_segmentation_v2.ipynb`, `eda_segmentation_final.ipynb`, hoặc `eda_customer.ipynb` chỉ để thử vài cells. Hãy cập nhật notebook hiện có, hoặc tạo script/function trong `src/eda/` rồi import vào notebook.

Notebook dùng để trình bày phải sạch. Trước khi push, kiểm tra:

```text
- Tên notebook đúng prefix và đúng intent.
- Markdown cells giải thích rõ mục tiêu phân tích.
- Code cells không chứa path tuyệt đối cá nhân nếu không cần thiết.
- Không có output quá nặng làm phình repo.
- Không có cell thử nghiệm rác, print debug vô nghĩa, hoặc commented code dài.
- Kết quả quan trọng được đưa vào docs hoặc exported assets nếu dùng cho report.
```

Notebook không phải nơi chứa business logic dài hạn. Nếu một function được dùng lại nhiều hơn một lần, chuyển nó vào `src/`.

## Quy định cho `src/`

`src/` chứa code có thể tái sử dụng. Mỗi module phải có intent rõ. Không ném tất cả vào root `src/`.

Nếu code thuộc EDA, đặt trong:

```text
src/eda/
```

Nếu code thuộc feature engineering:

```text
src/features/
```

Nếu code thuộc model training/inference:

```text
src/models/
```

Nếu code tạo visualization/report figures:

```text
src/visualization/
```

Nếu code dùng chung:

```text
src/utils/
```

Tên file Python phải là snake_case và nói rõ nhiệm vụ:

```text
src/eda/rfm_objective_segmentation.py
src/eda/customer_distribution_map.py
src/eda/segment_behavior_profiles.py
src/models/sales_forecasting.py
src/features/calendar_features.py
src/utils/io.py
```

Ví dụ không được chấp nhận:

```text
src/test.py
src/final.py
src/final_model_v5.py
src/new_model.py
src/abc.py
src/code.py
src/main2.py
src/lgbm2.py
```

Không tạo chuỗi version bằng tên file như `model_v1.py`, `model_v2.py`, `model_v3.py`. Versioning là việc của Git. Nếu cần nhiều approach, đặt tên theo approach:

```text
src/models/seasonal_baseline.py
src/models/gradient_boosting_sales.py
src/models/stacked_forecast.py
```

Mỗi file trong `src/` nên có docstring ngắn ở đầu file giải thích mục đích. Function quan trọng phải có docstring giải thích input, output, assumption và lý do business nếu liên quan đến EDA.

## Quy định về root directory

Root directory chỉ nên chứa project-level files:

```text
README.md
pyproject.toml
.python-version
.gitignore
main.py
```

Không đặt report, notebook, CSV output, image, PDF build artifact, hoặc temporary file ở root.

Không tạo folder mới ở root nếu folder đó không thuộc cấu trúc đã thống nhất. Nếu thật sự cần folder mới, phải nêu lý do trong PR hoặc team chat trước khi push.

## Quy định về tên file

Tên file phải:

```text
- Dùng tiếng Anh nếu là code/data artifact.
- Dùng snake_case cho code, data, generated assets.
- Có prefix theo intent nếu là notebook.
- Không dùng tên mơ hồ như final, new, test, temp, result, output.
- Không dùng version number nếu Git đã đủ quản lý version.
- Không dùng tên người trong filename trừ khi đó là folder sandbox đã được thống nhất.
```

Ví dụ tốt:

```text
eda_customer_segmentation.ipynb
model_sales_forecasting.ipynb
rfm_objective_segmentation.py
customer_ltv_segment_summary.csv
promotion_margin_leakage.png
```

Ví dụ xấu:

```text
eda_v4.ipynb
main_vietnam.tex
result_final.csv
submission_final_kaggle.csv
report2.pdf
new_analysis.py
test_plot.png
```

## Quy định về generated artifacts

Không push generated artifacts trừ khi chúng phục vụ report hoặc submission.

Được push:

```text
docs/assets/<topic>/*.png
docs/assets/<topic>/*.csv
submission.csv nếu cần nộp hoặc audit
final report PDF nếu đúng deliverable
```

Không được push:

```text
*.aux
*.fdb_latexmk
*.fls
*.out
*.xdv
__pycache__/
.ipynb_checkpoints/
random HTML exports
debug plots
temporary metrics files
```

Nếu file có thể regenerate bằng script, ưu tiên push script thay vì push hàng loạt output.

## Commit message convention

Commit message phải theo format:

```text
<type>(<scope>): <short imperative summary>
```

Các `type` được phép:

```text
feat      # Thêm feature, notebook section, chart, model, report section mới
fix       # Sửa bug, sửa số liệu sai, sửa leakage, sửa path lỗi
docs      # Cập nhật documentation/report/markdown
refactor  # Đổi cấu trúc code nhưng không đổi behavior
chore     # Việc phụ trợ: cleanup, config, gitignore, dependency
test      # Thêm/sửa kiểm thử hoặc validation script
data      # Thêm/sửa processed data có chủ đích
model     # Thay đổi modeling pipeline hoặc forecast submission
eda       # Thêm/sửa phân tích EDA hoặc visualization
```

Ví dụ tốt:

```text
feat(eda): add objective-weighted customer segmentation
eda(segmentation): add quarterly cohort retention analysis
docs(report): expand profit-secured loyalty narrative
fix(model): remove leakage from sales forecasting features
refactor(src): move calendar utilities into utils module
data(processed): add customer golden table with final segments
```

Ví dụ không được chấp nhận:

```text
update
fix
final
push code
new results
done
abc
v2
latest
```

Commit phải nhỏ và có một ý nghĩa. Không gộp các việc không liên quan như sửa report, thêm model, đổi data, generate PDF và cleanup notebook trong cùng một commit.

## Branch naming convention

Branch phải có prefix theo mục tiêu:

```text
feat/<short-topic>
fix/<short-topic>
docs/<short-topic>
eda/<short-topic>
model/<short-topic>
chore/<short-topic>
```

Ví dụ:

```text
eda/customer-segmentation-ltv
model/sales-forecast-xgboost
docs/project-structure-rules
fix/cohort-retention-quarter
```

Không dùng branch:

```text
new
test
final
my-branch
truong
khanh
v3
```

## Pull request hoặc push checklist

Trước khi push, tự kiểm tra:

```text
- File mới có nằm đúng folder không?
- Tên file có nói rõ intent không?
- Có tạo folder root mới không? Nếu có, tại sao?
- Có notebook trùng intent với notebook hiện có không?
- Có output/debug/temp artifact bị push nhầm không?
- Markdown có nằm trong docs không?
- Plot/report assets có nằm trong docs/assets/<topic>/ không?
- Code tái sử dụng đã chuyển vào src chưa?
- Commit message có đúng format không?
- Có sửa nhầm file của người khác hoặc xóa output họ đang dùng không?
```

Nếu câu trả lời cho bất kỳ câu nào là “không chắc”, dừng lại và hỏi team trước khi push.

## Migration rule cho repo hiện tại

Repo hiện tại đã có một số folder và file chưa theo chuẩn. Không xóa ngay lập tức nếu chưa biết ai đang dùng. Nhưng từ giờ:

```text
- Không thêm file mới vào results_*.
- Không thêm report mới vào root-level report/ nếu chưa thống nhất.
- Không tạo notebook mới trong results_*.
- Không tạo thêm file kiểu *_v1, *_v2, final_final.
- Nếu cần dùng lại artifact cũ, migrate vào docs/, notebooks/, src/ hoặc data/processed/ với tên rõ ràng.
```

Khi migrate, dùng commit riêng:

```text
refactor(repo): migrate scattered report artifacts into docs
refactor(notebooks): consolidate duplicated eda notebooks
chore(repo): remove obsolete generated latex artifacts
```

Không migrate bằng cách copy thêm một bản mới rồi để bản cũ nằm đó mãi. Migrate nghĩa là di chuyển có chủ đích, cập nhật reference, và xóa bản thừa nếu an toàn.

## Quy tắc cuối cùng

Nếu file không trả lời được ba câu hỏi dưới đây, không push:

```text
1. File này phục vụ mục tiêu nào của project?
2. Người khác nhìn tên file có hiểu nó làm gì không?
3. File này có nằm đúng nơi mà team kỳ vọng sẽ tìm nó không?
```

Repository sạch không phải để đẹp. Repository sạch giúp team chạy nhanh hơn, giảm conflict, giảm mất thời gian tìm file, và tăng khả năng tái lập kết quả khi nộp bài.
