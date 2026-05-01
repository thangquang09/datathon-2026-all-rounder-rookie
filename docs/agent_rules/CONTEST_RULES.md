# Nội dung Đề thi DATATHON 2026 - The Gridbreakers (Phần 2 & 3)

## Phần 2 — Trực quan hoá và Phân tích Dữ liệu (60 điểm)

Khám phá bộ dữ liệu để tìm ra các insight có ý nghĩa kinh doanh. Phần này được đánh giá dựa trên tính sáng tạo, chiều sâu phân tích và chất lượng trình bày. Không có đáp án đúng duy nhất — ban giám khảo đánh giá khả năng kể chuyện bằng dữ liệu (data storytelling) của các đội.

### Yêu cầu
Các đội thi tự do lựa chọn góc nhìn phân tích từ bộ dữ liệu. Bài nộp cần bao gồm hai thành phần:
1. **Trực quan hoá (Visualizations)**: Tạo các biểu đồ, đồ thị, bản đồ hoặc dashboard trực quan để thể hiện các pattern, xu hướng và mối quan hệ trong dữ liệu. Mỗi hình ảnh cần có tiêu đề, nhãn trục rõ ràng và chú thích phù hợp.
2. **Phân tích (Analysis)**: Viết phần giải thích đi kèm mỗi trực quan hoá, bao gồm:
   - Mô tả những gì biểu đồ thể hiện và tại sao góc nhìn này quan trọng
   - Các phát hiện chính (key findings) được hỗ trợ bởi số liệu cụ thể
   - Ý nghĩa kinh doanh (business implications) hoặc đề xuất hành động (actionable recommendations)

### Tiêu chí đánh giá Phần 2
Bài nộp được đánh giá theo bốn cấp độ phân tích. Cấp độ cao hơn bao gồm và nâng cao cấp độ thấp hơn.
- **Descriptive (What happened?)**: Thống kê tổng hợp chính xác, biểu đồ có nhãn rõ ràng, tổng hợp dữ liệu đúng.
- **Diagnostic (Why did it happen?)**: Giả thuyết nhân quả, so sánh phân khúc, xác định bất thường có bằng chứng hỗ trợ.
- **Predictive (What is likely to happen?)**: Ngoại suy xu hướng, phân tích tính mùa vụ, phân tích chỉ số dẫn xuất.
- **Prescriptive (What should we do?)**: Đề xuất hành động kinh doanh được hỗ trợ bởi dữ liệu; đánh đổi được định lượng.
*Lưu ý: Các đội đạt cấp độ Prescriptive nhất quán trên nhiều phân tích sẽ đạt điểm cao nhất.*

#### Thang điểm chi tiết (Tổng 60 điểm)
Phần này được chấm theo bốn tiêu chí độc lập, ban giám khảo chấm từng tiêu chí trên thang điểm thành phần, sau đó cộng lại:

1. **Chất lượng trực quan hoá (Tối đa 15 điểm)**
   - **13–15đ**: Tất cả biểu đồ đều đạt chuẩn (có tiêu đề, nhãn trục, chú thích đầy đủ), lựa chọn loại biểu đồ tối ưu cho từng insight.
   - **8–12đ**: Phần lớn biểu đồ đạt yêu cầu, một số thiếu nhãn hoặc chú thích.
   - **0–7đ**: Biểu đồ thiếu thông tin, khó đọc hoặc không phù hợp với dữ liệu.

2. **Chiều sâu phân tích (Tối đa 25 điểm)**
   - **21–25đ**: Bao phủ và đạt cả bốn cấp độ Descriptive, Diagnostic, Predictive, Prescriptive một cách nhất quán. Lập luận logic, có số liệu cụ thể hỗ trợ.
   - **14–20đ**: Đạt ba cấp độ, cấp độ Prescriptive còn hời hợt.
   - **7–13đ**: Chủ yếu ở cấp Descriptive và Diagnostic.
   - **0–6đ**: Chỉ mô tả bề mặt, thiếu phân tích.

3. **Insight kinh doanh (Tối đa 15 điểm)**
   - **13–15đ**: Phát hiện có giá trị thực tiễn; đề xuất cụ thể, định lượng được, áp dụng được ngay. Liên kết rõ ràng giữa dữ liệu và quyết định kinh doanh.
   - **8–12đ**: Có đề xuất nhưng còn chung chung.
   - **0–7đ**: Thiếu kết nối với bối cảnh kinh doanh.

4. **Tính sáng tạo & kể chuyện (Tối đa 5 điểm)**
   - **4–5đ**: Góc nhìn độc đáo, không lặp lại các phân tích hiển nhiên; kết hợp nhiều nguồn dữ liệu, mạch trình bày coherent, thuyết phục.
   - **2–3đ**: Có điểm sáng tạo nhưng chưa nhất quán.
   - **0–1đ**: Phân tích dự đoán được, không có điểm nổi bật.

---

## Phần 3 — Mô hình Dự báo Doanh thu (Sales Forecasting) (20 điểm)

### Bối cảnh kinh doanh
Bạn là nhà khoa học dữ liệu tại một công ty thương mại điện tử thời trang Việt Nam. Doanh nghiệp cần dự báo nhu cầu chính xác ở mức chi tiết để tối ưu hoá phân bổ tồn kho, lập kế hoạch khuyến mãi và quản lý logistics trên toàn quốc.

### Định nghĩa bài toán
Dự báo cột `Revenue` trong khoảng thời gian của `sales_test.csv`.
Mỗi dòng trong tập test là một bộ `(Date, Revenue, COGS)` duy nhất trong giai đoạn `01/01/2023 – 01/07/2024`.

### Dữ liệu
| Split | File | Khoảng thời gian |
| --- | --- | --- |
| **Train** | `sales.csv` | `04/07/2012 – 31/12/2022` |
| **Test** | `sales_test.csv` | `01/01/2023 – 01/07/2024` |

### Chỉ số đánh giá
Bài nộp được đánh giá bằng ba chỉ số:
- **Mean Absolute Error (MAE)**: Đo độ lệch tuyệt đối trung bình. Phấn đấu đạt mức càng thấp càng tốt.
- **Root Mean Squared Error (RMSE)**: Phạt nặng hơn các sai số lớn. Phấn đấu đạt mức càng thấp càng tốt.
- **R² (Coefficient of Determination)**: Thể hiện tỷ lệ phương sai được giải thích bởi mô hình. Phấn đấu đạt mức càng cao càng tốt (lý tưởng gần 1).

### Định dạng file nộp
Nộp file `submission.csv` với các cột sau:
Các dòng trong `submission.csv` phải giữ đúng thứ tự như `sample_submission.csv`. **Không sắp xếp lại hoặc xáo trộn.**
Ví dụ:
```csv
Date,Revenue,COGS
2023-01-01,26607.2,2585.15
2023-01-02,1007.89,163.0
2023-01-03,1089.51,821.12
...
```

### Ràng buộc & Điều kiện loại bài
1. **Không dùng dữ liệu ngoài**: Tất cả đặc trưng phải được tạo từ các file dữ liệu được cung cấp.
2. **Tính tái lập (Reproducibility)**: Đính kèm toàn bộ mã nguồn. Đặt random seed khi cần thiết.
3. **Khả năng giải thích (Explainability)**: Trong report, bao gồm một mục giải thích các yếu tố dẫn động doanh thu chính được mô hình xác định (vd: feature importances, SHAP values, hoặc partial dependence plots). Giải thích những gì mô hình học được bằng ngôn ngữ kinh doanh.

**⚠️ Điều kiện loại bài (Loại toàn bộ Phần 3 nếu vi phạm)**:
- Sử dụng Revenue/COGS từ tập test làm đặc trưng (Data Leakage).
- Sử dụng dữ liệu ngoài bộ dữ liệu được cung cấp.
- Không đính kèm mã nguồn hoặc kết quả không thể tái lập.

#### Thang điểm chi tiết (Tổng 20 điểm)
Điểm Phần 3 được tính từ hai thành phần: hiệu suất mô hình trên Kaggle và chất lượng báo cáo kỹ thuật.

1. **Hiệu suất mô hình (Tối đa 12 điểm)**
   - **10–12đ**: Xếp hạng top leaderboard (Dựa trên điểm MAE, RMSE, R² trên tập test); MAE và RMSE thấp, R² cao.
   - **5–9đ**: Hiệu suất trung bình; mô hình hoạt động nhưng chưa tối ưu.
   - **3–4đ**: Bài nộp hợp lệ nhưng hiệu suất thấp; mức điểm sàn.

2. **Báo cáo kỹ thuật (Tối đa 8 điểm)**
   - **7–8đ**: Pipeline rõ ràng, cross-validation đúng chiều thời gian, **giải thích mô hình cụ thể bằng SHAP hoặc tương đương**, tuân thủ đầy đủ ràng buộc.
   - **4–6đ**: Pipeline đủ dùng, giải thích còn định tính, một số ràng buộc chưa được xử lý tường minh.
   - **0–3đ**: Thiếu giải thích, không kiểm soát leakage, hoặc không thể tái lập kết quả.
