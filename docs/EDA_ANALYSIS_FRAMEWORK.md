# Phần 2 - Trực quan hóa & Phân tích EDA

Tài liệu này là khung làm bài để nhắm tối đa 60 điểm ở Phần 2. Mục tiêu không phải vẽ nhiều biểu đồ, mà là xây dựng một câu chuyện dữ liệu rõ ràng, có số liệu và có ý nghĩa kinh doanh.

## 1. Mục tiêu chấm điểm

Phần 2 được chấm theo 4 tiêu chí độc lập:

| Tiêu chí | Điểm | Bản chất |
|---|---:|---|
| Chất lượng trực quan hóa | 15 | Biểu đồ đúng loại, có tiêu đề, nhãn trục, chú thích, dễ đọc |
| Chiều sâu phân tích | 25 | Đi từ mô tả → chẩn đoán → dự báo → đề xuất |
| Insight kinh doanh | 15 | Có giá trị thực tiễn, gắn với quyết định kinh doanh |
| Sáng tạo & kể chuyện | 5 | Góc nhìn độc đáo, kết nối nhiều bảng có chủ đích |

## 2. Khung narrative để đạt điểm cao

Nên viết report theo mạch sau:

1. Tổng quan hệ sinh thái dữ liệu và các câu hỏi kinh doanh chính.
2. Mô tả các xu hướng lớn và phân bố cơ bản.
3. Chẩn đoán nguyên nhân bằng cách cắt lớp theo segment, region, channel, product, promo, time.
4. Dự đoán xu hướng tiếp theo từ pattern lịch sử, mùa vụ, động lực tăng trưởng.
5. Đề xuất hành động cụ thể có thể áp dụng ngay.

## 3. Các nhóm phân tích nên có

### 3.1 Doanh thu và biên lợi nhuận theo thời gian

Nên có:

- Biểu đồ đường doanh thu theo ngày/tháng.
- Mẫu mùa vụ theo tháng, quý, thứ trong tuần, dịp lễ.
- Nếu có thể, so sánh Revenue với COGS hoặc gross margin.

Nội dung cần viết:

- Có xu hướng tăng hay giảm?
- Có mùa vụ rõ ràng không?
- Bằng chứng nào cho thấy một giai đoạn bất thường?
- Nếu duy trì xu hướng này, kỳ tiếp theo sẽ ra sao?
- Nên làm gì về tồn kho, khuyến mãi hoặc logistics?

### 3.2 Khách hàng và hành vi mua

Nên có:

- Phân bố theo age_group, gender, acquisition_channel.
- So sánh số đơn/khách và giá trị đơn theo nhóm.
- Cohort hoặc repeat purchase nếu dữ liệu cho phép.

Nội dung cần viết:

- Nhóm khách nào mang lại giá trị cao nhất?
- Nhóm nào có tần suất mua cao nhưng giá trị đơn nhỏ?
- Kênh nào thu hút khách tốt nhưng chất lượng thấp, hoặc ngược lại?
- Nên ưu tiên nguồn lực marketing vào phân khúc nào?

### 3.3 Sản phẩm, segment, category

Nên có:

- Top/bottom category theo doanh thu, biên lợi nhuận, tỷ lệ trả hàng.
- Phân bố giá và COGS theo segment.
- Phân tích chéo giữa size, color, category, return rate.

Nội dung cần viết:

- Sản phẩm/segment nào đang kéo doanh thu nhiều nhất?
- Sản phẩm nào có biên lợi nhuận tốt nhưng tỷ lệ trả cao?
- Có mẫu hình nào cho thấy một size hoặc segment bị vấn đề chất lượng/expectation?
- Nếu cắt giảm một nhóm sản phẩm, tác động có thể là gì?

### 3.4 Khuyến mãi và hiệu quả giảm giá

Nên có:

- Tỷ lệ đơn có promo.
- So sánh doanh thu, đơn giá, biên lợi nhuận khi có/không có promo.
- Hiệu quả theo promo_type, promo_channel, stackable_flag, min_order_value.

Nội dung cần viết:

- Khuyến mãi có thật sự kéo doanh thu hay chỉ ăn biên lợi nhuận?
- Loại khuyến mãi nào hiệu quả hơn?
- Có cơ hội tối ưu điều kiện áp dụng để tăng AOV hoặc margin không?

### 3.5 Geography, logistics, vận hành

Nên có:

- Biểu đồ theo region, city, district, zip.
- Kết hợp orders, shipments, returns, reviews.
- Nếu có thể, đối chiếu với inventory và stockout.

Nội dung cần viết:

- Vùng nào có doanh thu cao, vùng nào có vấn đề vận hành?
- Tỷ lệ giao trễ, trả hàng hoặc đánh giá xấu có tập trung theo khu vực không?
- Tồn kho có phản ánh được nhu cầu thực tế không?

### 3.6 Web traffic và phao đo tìm kiếm cơ hội

Nên có:

- Sessions, unique visitors, bounce rate, page views theo traffic source.
- So sánh traffic với đơn hàng/doanh thu nếu có thể liên kết.

Nội dung cần viết:

- Nguồn traffic nào chất lượng tốt nhất?
- Bounce rate có liên hệ với doanh thu hay chuyển đổi không?
- Nên dồn ngân sách sang kênh nào?

## 4. Mẫu khung cho mỗi biểu đồ

Mỗi biểu đồ nên có 4 phần viết kèm:

1. Biểu đồ đang nói gì.
2. Tại sao góc nhìn này quan trọng.
3. Key findings có số liệu cụ thể.
4. Business implication hoặc hành động đề xuất.

Mẫu đoạn viết:

"Biểu đồ này cho thấy ..."

"Điểm nổi bật là ... với mức chênh lệch ..."

"Điều này gợi ý rằng ..."

"Về mặt kinh doanh, cần ..."

## 5. Mapping sang 4 cấp độ phân tích

### Descriptive

- Nêu ra con số, xu hướng, phân bố.
- Trả lời câu hỏi: đã xảy ra cái gì?

### Diagnostic

- Cắt theo segment, region, channel, size, promo, time.
- Trả lời câu hỏi: vì sao xảy ra?

### Predictive

- Ngoại suy xu hướng, nhận diện mùa vụ, forecast ngắn hạn.
- Trả lời câu hỏi: sẽ xảy ra gì tiếp theo?

### Prescriptive

- Đề xuất quyết định có thể hành động được.
- Trả lời câu hỏi: nên làm gì?

## 6. Checklist để đạt 60/60

- Mỗi biểu đồ có tiêu đề, nhãn trục, đơn vị, chú thích nếu cần.
- Không dùng biểu đồ chỉ để mô tả; phải có ý nghĩa kinh doanh.
- Ít nhất một phần có liên kết nhiều bảng dữ liệu.
- Mỗi insight có số liệu hỗ trợ rõ ràng.
- Có ít nhất một đoạn dự đoán và một đoạn đề xuất hành động.
- Không lặp lại cùng một ý tưởng ở nhiều biểu đồ.
- Câu chuyện tổng thể phải chạy từ dữ liệu → vấn đề → hành động.

## 7. Logic Tree MECE cho mục tiêu tối ưu doanh thu và trải nghiệm khách hàng

Mục tiêu tổng:

- Tăng doanh thu bền vững (không đánh đổi trải nghiệm khách hàng).

### 7.1 Cây 1 - Doanh thu thuần (Net Revenue) theo MECE

Phân rã theo công thức để đảm bảo các nhánh không chồng lắp:

$$
NetRevenue = GMV - Discount - Refund
$$

$$
GMV = Sessions \times CVR \times AOV
$$

Trong đó:

- Sessions: lưu lượng truy cập.
- CVR (conversion rate): tỷ lệ chuyển đổi từ phiên sang đơn.
- AOV (average order value): giá trị đơn trung bình.

Logic tree:

```text
Mục tiêu: Tăng Net Revenue
├─ A. Tăng GMV
│  ├─ A1. Tăng Sessions
│  │  ├─ Nguồn traffic chất lượng cao
│  │  └─ Giảm bounce, tăng độ sâu phiên
│  ├─ A2. Tăng CVR
│  │  ├─ Tối ưu theo device_type / order_source
│  │  └─ Giảm drop do stockout hoặc thanh toán
│  └─ A3. Tăng AOV
│     ├─ Tăng items per order (cross-sell, bundle)
│     └─ Tối ưu net unit price theo promo mix
├─ B. Giảm Discount Leakage
│  ├─ B1. Giảm discount không hiệu quả theo promo_type
│  ├─ B2. Hạn chế stackable promo gây bào mòn doanh thu
│  └─ B3. Kiểm soát min_order_value để bảo vệ AOV
└─ C. Giảm Refund Leakage
	├─ C1. Giảm return rate theo category/size/color
	├─ C2. Giảm refund per returned item
	└─ C3. Giảm return do giao hàng chậm hoặc sai kỳ vọng
```

Mapping dữ liệu theo nhánh:

- A1 (Sessions): `web_traffic.csv` (sessions, unique_visitors, bounce_rate, traffic_source).
- A2 (CVR): `web_traffic.csv` + `orders.csv` (orders/date, order_source, device_type).
- A3 (AOV): `orders.csv` + `order_items.csv` + `payments.csv`.
- B (Discount): `order_items.csv` + `promotions.csv`.
- C (Refund): `returns.csv` + `order_items.csv` + `orders.csv` + `shipments.csv`.

### 7.2 Cây 2 - Trải nghiệm khách hàng (CX) theo MECE

Phân rã theo hành trình khách hàng, mỗi nhánh là một giai đoạn riêng biệt:

```text
Mục tiêu: Nâng CX tổng thể
├─ D. Trải nghiệm trước mua (Discovery Experience)
│  ├─ D1. Chất lượng traffic (bounce rate, session duration)
│  └─ D2. Mức độ phù hợp kênh-thông điệp theo traffic_source
├─ E. Trải nghiệm mua hàng (Purchase Experience)
│  ├─ E1. Tỷ lệ hoàn tất đơn theo device_type/payment_method
│  └─ E2. Ma sát do giá/khuyến mãi/phương thức thanh toán
├─ F. Trải nghiệm giao nhận (Delivery Experience)
│  ├─ F1. Lead time giao hàng
│  └─ F2. Tính ổn định theo region/city/zip
├─ G. Trải nghiệm sản phẩm (Product Experience)
│  ├─ G1. Rating trung bình theo product/category/segment
│  └─ G2. Return reason và mismatch kỳ vọng
└─ H. Trải nghiệm sau mua (Post-sale Experience)
	├─ H1. Tỷ lệ hoàn trả và hoàn tiền
	└─ H2. Khả năng giữ chân (repeat behavior theo customer)
```

Mapping dữ liệu theo nhánh:

- D: `web_traffic.csv`.
- E: `orders.csv` + `payments.csv` + `order_items.csv`.
- F: `shipments.csv` + `orders.csv` + `geography.csv`.
- G: `reviews.csv` + `returns.csv` + `products.csv`.
- H: `returns.csv` + `orders.csv` + `customers.csv`.

### 7.3 Bộ KPI khuyến nghị theo từng nhánh

- Sessions, Bounce Rate, Avg Session Duration, Pages/Session.
- CVR = số order / số session.
- AOV = tổng payment_value / số order.
- Discount Rate = tổng discount_amount / tổng (quantity * unit_price + discount_amount).
- Return Rate = số lượng trả / số lượng bán.
- Refund Rate = tổng refund_amount / Net Revenue.
- Delivery Lead Time = delivery_date - order_date.
- Review Score = rating trung bình và tỷ trọng rating <= 2.
- Repeat Purchase Rate = số khách có >= 2 đơn / tổng khách.

### 7.4 Ưu tiên phân tích theo ma trận tác động

Nên ưu tiên theo thứ tự:

1. Nhánh có tác động tài chính cao và làm xấu CX cùng lúc (ví dụ: giao chậm -> return tăng).
2. Nhánh có thể can thiệp nhanh bằng vận hành hoặc promo policy.
3. Nhánh cần thay đổi dài hạn về danh mục sản phẩm hoặc chiến lược kênh.
