# Datathon 2026 - EDA Insight Report (Part 2)

## 1. Mục tiêu phân tích

Báo cáo khám phá ba câu hỏi kinh doanh cốt lõi từ dữ liệu e-commerce thời trang Việt Nam (2012–2022):

1. **Sản phẩm nào đang gây tổn hại đến thương hiệu?** — nhóm bị trả lại nhiều và review kém
2. **Sản phẩm nào đang "ngủ quên" trong kho?** — nhóm chưa phát sinh đơn hàng
3. **Sản phẩm nào thường được mua cùng nhau?** — cơ hội cross-sell và bundle

---

## 2. Tóm tắt kết quả

| Câu hỏi | Kết quả chính |
|---|---|
| Sản phẩm rủi ro | **6 SKU** vừa có return rate cao vừa có rating thấp |
| Sản phẩm chưa bán | **814 SKU** chưa có đơn hàng nào trong toàn bộ giai đoạn |
| Cặp mua kèm mạnh | **55 cặp mua kèm duy nhất** sau khi gộp rule hai chiều; top 20 cặp chiếm khoảng **35.8K đơn mua kèm** |

---

## 3. Chi tiết phân tích

### 3.1. Nhóm sản phẩm bị trả lại nhiều và review kém

**Tiêu chí xác định sản phẩm rủi ro:**
- Đã bán ít nhất 100 sản phẩm (`sold_qty ≥ 100`)
- Có ít nhất 10 lượt đánh giá (`review_count ≥ 10`)
- Điểm đánh giá trung bình ≤ 3.5/5
- Tỷ lệ trả hàng ≥ 5%

**Lý giải từng tiêu chí:**

**`sold_qty ≥ 100`**

Tiêu chí này không chọn 100 tùy tiện. *Tín hiệu cần phát hiện* được định nghĩa trước: sản phẩm có return rate cao hơn baseline ít nhất **1.5 điểm phần trăm** — tức ≥ 5%, khoảng 40% cao hơn mặt bằng platform 3.41%. Để chọn n tối thiểu, sai số đo lường (MAE của ước tính return rate) phải nhỏ hơn 1.5pp này.

Kết quả Parametric Bootstrap (873 sản phẩm làm ground truth, 500 lần giả lập mỗi điểm):

| sold_qty | Sai số ước tính return rate (MAE) | So với tín hiệu 1.5pp |
|---:|---:|---|
| 20 | 3.26pp | Noise gấp 2.2× signal |
| 50 | 2.01pp | Noise gấp 1.3× signal — không phân tách được |
| 75 | 1.64pp | Noise vẫn lớn hơn signal |
| **100** | **1.42pp** | **Noise < signal — lần đầu tiên có thể phân tách** ✓ |
| 120 | 1.30pp | Tiếp tục cải thiện |
| 200 | 1.00pp | Cải thiện biên giảm dần |

Ở `n=75`, nhiễu (1.64pp) vẫn lớn hơn tín hiệu (1.5pp). Ở `n=100`, lần đầu tiên nhiễu xuống dưới tín hiệu — ngưỡng tối thiểu để ước tính return rate đủ tin cậy phân biệt sản phẩm rủi ro khỏi biến động ngẫu nhiên.

**`avg_rating ≤ 3.5`**

Rating trung bình toàn platform là **3.94**; trong nhóm 891 sản phẩm đủ điều kiện, mức trung bình là **3.93** và P25 là **3.84**. Ngưỡng 3.5 tương đương *bottom 3–5%* — cách mặt bằng chung 0.44 điểm trên thang 5, đủ có ý nghĩa thực tế.

**`return_rate ≥ 5%`**

Return rate trung bình toàn platform là 3.41% và P90 là 4.71%. Ngưỡng 5% rơi vào **top 7.4% sản phẩm có tỷ lệ trả hàng cao nhất** — vượt P90, tức thực sự bất thường, không chỉ nhỉnh hơn bình thường. Ngưỡng này cũng nhất quán với tín hiệu 1.5pp đã dùng để chọn `sold_qty ≥ 100`: cả hai quy về cùng mức ≥ 5%.

**`review_count ≥ 10`**

Non-parametric Bootstrap (441 sản phẩm làm ground truth, 200 lần subsampling mỗi điểm) cho thấy:

| review_count | Sai số avg_rating (MAE) | Tỷ lệ phân loại sai tại ngưỡng 3.5 |
|---:|---:|---:|
| 5 | 0.404 điểm | 19.4% |
| **10** | **0.278 điểm** | **14.8% — 1 trong 7 sản phẩm bị phân loại sai** |
| 15 | 0.221 điểm | 7.6% — cải thiện gấp đôi so với n=10 |
| 20 | 0.185 điểm | 6.0% |

Ngưỡng chặt hơn về mặt thống kê là `review_count ≥ 15` (tỷ lệ phân loại sai giảm một nửa). Với `return_rate ≥ 5%`, 5/6 sản phẩm rủi ro đã có `review_count ≥ 13`, nên tác động của ngưỡng này lên kết quả là nhỏ. Ngưỡng 10 là lựa chọn thực dụng — nhưng kết luận về rating với dưới 15 reviews mang độ tin cậy thấp hơn.

---

**Kết quả: 6 sản phẩm thuộc nhóm rủi ro cao.**

Toàn bộ 6 sản phẩm (xếp theo problem score):

Problem score = tổng z-score của ba chiều so với nhóm 891 sản phẩm đủ điều kiện: `z(return_rate) + z(−avg_rating) + z(low_rating_share)`. Mỗi chiều đóng góp ngang nhau; giá trị dương nghĩa là lệch về phía tệ hơn mặt bằng.

| Sản phẩm | Danh mục | Phân khúc | Tỷ lệ trả hàng | Điểm TB | Số lượng bán | Số lượng trả | Problem score |
|---|---|---|---:|---:|---:|---:|---:|
| SaigonCore YY-50 | GenZ | Trendy | 6.68% | 3.38 | 479 | 32 | 8.36 |
| HanoiStreet UC-37 | Streetwear | Everyday | 6.70% | 3.32 | 701 | 47 | 7.02 |
| SaigonFlex UC-34 | Streetwear | Everyday | 6.86% | 3.44 | 1,166 | 80 | 6.04 |
| MekongFit UC-16 | Streetwear | Everyday | 5.31% | 3.38 | 471 | 25 | 5.81 |
| LotusWear UR-18 | Streetwear | Standard | 8.03% | 3.50 | 299 | 24 | 5.41 |
| MekongFit RP-33 | Outdoor | Activewear | 6.87% | 3.44 | 626 | 43 | 5.14 |

**LotusWear UR-18** có return rate cao nhất nhóm (8.03%) — cứ ~12.5 đơn lại có 1 đơn bị trả. **SaigonFlex UC-34** là trường hợp nặng nhất về quy mô: bán hơn 1,000 sản phẩm và tạo ra khoản hoàn tiền lớn nhất nhóm.

**Tác động business của 6 sản phẩm rủi ro:**

| Chỉ số | Nhóm 6 SKU | Toàn platform |
|---|---:|---:|
| Doanh thu | 31.0M VND | 15,681M VND (chiếm **0.20%**) |
| Hoàn tiền khách hàng | **2.05M VND** | 510.6M VND (chiếm **0.40%**) |
| Tỷ lệ hoàn/doanh thu | **6.61%** | 3.26% |
| Số SKU | 6 (0.38% SKU đã bán) | 1,598 SKU |

6 SKU này chỉ chiếm 0.20% doanh thu nhưng **0.40% tổng hoàn tiền** — tỷ lệ hoàn/doanh thu (6.61%) cao hơn platform **gấp đôi** (3.26%). Hoàn tiền lớn nhất: SaigonFlex UC-34 (1.15M VND), HanoiStreet UC-37 (315K VND), LotusWear UR-18 (191K VND — tỷ lệ hoàn/doanh thu cao nhất nhóm: **8.02%**).

~2M VND hoàn tiền nhỏ về tuyệt đối, nhưng tỷ lệ hoàn/doanh thu gấp đôi bình quân platform — nhóm này tạo chi phí reverse logistics và customer service không cân xứng. Tác động gián tiếp qua churn (xem mục 3.5) có thể lớn hơn nhiều.

![Top problematic products](docs/assets/eda_insight_report/figures/problematic_products_top15.png)

---

### 3.2. Phân tích lý do trả hàng

Toàn hệ thống có **39,939 lượt trả hàng** với 5 lý do chính:

| Lý do | Tỷ lệ theo số lượt | Tỷ lệ theo giá trị hoàn |
|---|---:|---:|
| Sai size | 34.97% | 34.60% |
| Sản phẩm lỗi | 20.08% | 20.29% |
| Không đúng mô tả | 17.61% | 17.75% |
| Đổi ý | 17.35% | 17.57% |
| Giao hàng trễ | 9.98% | 9.78% |

So sánh nhóm 6 sản phẩm rủi ro với mặt bằng chung:

| Lý do | Toàn hệ thống | Nhóm rủi ro | Chênh lệch |
|---|---:|---:|---:|
| Giao hàng trễ | 9.83% | 20.32% | **+10.49 điểm %** |
| Đổi ý | 17.47% | 22.71% | **+5.24 điểm %** |
| Sản phẩm lỗi | 20.33% | 7.97% | −12.36 điểm % |

**Nhận xét:** Nhóm 6 sản phẩm rủi ro bị trả do giao hàng trễ nhiều hơn **gấp đôi** bình quân platform (20.3% so với 9.8%). Tuy nhiên, phân tích thời gian giao hàng thực tế (xem mục 3.6) cho thấy đây **không phải vấn đề fulfillment**: thời gian giao hàng của các đơn trả với lý do `late_delivery` từ nhóm rủi ro có median **6 ngày** — bằng hệt mặt bằng platform. Khách hàng có khả năng đang dùng lý do "giao hàng trễ" như một lý do thay thế cho sự không hài lòng về sản phẩm.

![Return reasons overall](docs/assets/eda_insight_report/figures/return_reason_overall.png)

![Return reasons all vs problematic](docs/assets/eda_insight_report/figures/return_reason_all_vs_problematic.png)

---

### 3.3. Nhóm sản phẩm chưa được mua

**Định nghĩa:** Sản phẩm có trong catalog nhưng không có đơn hàng nào trong toàn bộ giai đoạn dữ liệu (2012–2022).

**Kết quả: 814 SKU chưa phát sinh doanh thu.**

Phân bố theo danh mục:

| Danh mục | Số SKU chưa bán | Tỷ lệ |
|---|---:|---:|
| Streetwear | 443 | 54.4% |
| Outdoor | 247 | 30.3% |
| Casual | 87 | 10.7% |
| GenZ | 37 | 4.5% |

Top 5 thương hiệu có nhiều SKU chưa bán nhất:

| Thương hiệu | SKU chưa bán |
|---|---:|
| VietMode | 154 |
| HanoiStreet | 150 |
| VietMotion | 113 |
| DragonWear | 96 |
| UrbanVN | 79 |

**Nhận xét:** Streetwear chiếm hơn 54% tổng SKU chưa bán. Tuy nhiên, cần diễn giải thận trọng: "chưa được mua" chỉ khẳng định SKU có trong `products.csv` nhưng không xuất hiện trong `order_items.csv`; dữ liệu hiện tại chưa cho biết các SKU này có từng được bật bán công khai, có traffic, hay có trạng thái active/inactive trong từng thời điểm hay không.

Vì vậy, không nên hiểu 814 SKU này là "hàng tồn 10 năm không bán được". Cách hiểu hợp lý hơn là: đây là các SKU nằm trong product master/catalog nhưng chưa tạo giao dịch trong dữ liệu bán hàng. Nguyên nhân có thể đến từ catalog lưu dư, SKU được tạo trước nhưng chưa launch, listing đã inactive trước khi có đơn, biến thể sản phẩm không được nhập hàng, hoặc sản phẩm chỉ tồn tại như bản ghi tham chiếu trong hệ thống.

![Unsold distribution](docs/assets/eda_insight_report/figures/unsold_distribution.png)

---

### 3.4. Sản phẩm thường được mua cùng nhau

Phân tích dùng hai cách tiếp cận bổ trợ nhau: **co-purchase count** (đếm trực tiếp, toàn catalog) để xác định volume cơ hội, và **Apriori** (thuật toán khai thác luật kết hợp, chỉ sản phẩm ≥ 200 đơn) để xác nhận tính thống kê và sinh rule có hướng cho recommendation engine.

**Thiết lập Apriori:** `min_support ≈ 0.000309`, `confidence ≥ 0.2`, `lift ≥ 1.2`. Apriori sinh rule có hướng (`A → B` và `B → A`), nên cùng một cặp xuất hiện hai dòng — hữu ích cho recommendation theo sản phẩm đang xem, nhưng cần dedup khi dùng cho bundle.

**Chỉ số đo lường:**

| Chỉ số | Ý nghĩa | Cách đọc trong business |
|---|---|---|
| `co_orders` | Số đơn hàng có cả hai sản phẩm | Đo quy mô cơ hội tuyệt đối |
| `support` | `P(A ∩ B)` — tỷ lệ trên tổng đơn | Support cao → campaign có nhiều traffic |
| `confidence A→B` | `P(B|A)` — xác suất mua B khi đã mua A | 0.50 nghĩa là cứ 2 khách mua A thì 1 khách mua B |
| `lift` | `confidence / P(B)` — mức liên kết vượt ngẫu nhiên | > 1 là liên kết dương; lift cao + volume thấp → chỉ phù hợp recommendation hẹp |
| `leverage` | Tần suất mua cùng thực tế − kỳ vọng nếu độc lập | Dương → cặp đi cùng nhiều hơn ngẫu nhiên |
| `conviction` | Đo mức "đáng tin" theo hướng A → B | Dùng phụ trợ, không dùng một mình để chọn campaign |

**Kết quả tổng quan:** 602 frequent itemsets → 110 association rules → **55 cặp sản phẩm duy nhất** sau khi dedup.

**Top 5 cặp theo volume (deduped):**

| Cặp sản phẩm | Số đơn mua kèm | Support | Confidence cao nhất | Lift |
|---|---:|---:|---:|---:|
| HanoiStreet RP-79 + HanoiStreet RP-80 | 5,158 | 0.00797 | 0.509 | 32.39 |
| HanoiStreet RP-21 + HanoiStreet RP-22 | 2,577 | 0.00398 | 0.512 | 65.54 |
| MekongFit RP-31 + MekongFit RP-32 | 2,465 | 0.00381 | 0.500 | 65.56 |
| HanoiStreet RP-47 + HanoiStreet RP-48 | 2,419 | 0.00374 | 0.505 | 67.91 |
| HanoiStreet RP-81 + HanoiStreet RP-82 | 2,255 | 0.00349 | 0.494 | 69.49 |

**Top 5 cặp theo lift (co_orders ≥ 50):**

| Cặp sản phẩm | Số đơn mua kèm | Lift |
|---|---:|---:|
| HanoiStreet RP-07 + HanoiStreet RP-08 | 53 | 4,329.30 |
| VietMode RP-89 + VietMode RP-90 | 55 | 3,630.81 |
| HanoiStreet RP-05 + HanoiStreet RP-06 | 51 | 3,543.95 |
| HanoiStreet RP-27 + HanoiStreet RP-28 | 61 | 3,355.75 |
| VietMode RP-99 + VietMode RP-00 | 68 | 3,080.69 |

**Insight business chính:** Các cặp mạnh đều là SKU liền dòng trong cùng thương hiệu — khách mua theo bộ, không phải ngẫu nhiên. Top 20 cặp dedup tạo khoảng **35.8K đơn mua kèm**, trong đó 13/20 thuộc HanoiStreet và 6/20 thuộc VietMotion.

**Sweet spot — 16 cặp** có co_orders > 1,000 và lift > 65: confidence ~0.50 nghĩa là cứ 2 khách mua một SKU thì 1 khách mua SKU liền dòng kèm — đây là nhóm ưu tiên cao nhất để thử "Add matching item" hoặc combo discount.

**Hai góc cần tránh khi chọn cặp để hành động:**
- **Lift cực cao, volume thấp** (`VietMotion RP-75 + RP-76`, lift ~910, ~207 đơn): chỉ phù hợp recommendation cá nhân hoá hẹp, không nên làm campaign đại trà.
- **Volume cực lớn, lift thấp** (`HanoiStreet RP-79 + RP-80`, 5,158 đơn, lift=32): khách đã tự mua kèm nhiều, bundle discount ít tác động biên.

![Top co-purchased pairs](docs/assets/eda_insight_report/figures/copurchase_top15_by_orders.png)

![Deduped Apriori top pairs by volume](docs/assets/eda_insight_report/figures/apriori_dedup_top15_volume.png)

![Lift vs co-orders](docs/assets/eda_insight_report/figures/copurchase_lift_scatter.png)

![Deduped Apriori lift vs volume](docs/assets/eda_insight_report/figures/apriori_dedup_lift_vs_volume.png)

---

### 3.5. Liên hệ với tỷ lệ rời bỏ khách hàng (Churn Linkage)

**Chỉ số đo lường churn:**
- `churn_180d`: khách không có đơn mới trong 180 ngày tính tới cuối kỳ dữ liệu
- `no-repeat_90d`: sau một sự kiện cụ thể, không có đơn mới trong 90 ngày tiếp theo

**Định nghĩa sự kiện:**
- *Trả hàng*: mỗi bản ghi trong `returns.csv`, dùng `return_date` làm mốc sự kiện
- *Review thấp*: mỗi review có `rating ≤ 2` (1–2 sao), dùng `review_date` làm mốc — **ngưỡng này khác với ngưỡng ≤ 3.5 dùng trong mục 3.1**; ở đây chỉ bắt nhóm cực kỳ bất mãn
- *Mua sản phẩm problematic*: đơn chứa bất kỳ sản phẩm nào trong top 17 theo `problem_score` (không chỉ 6 SKU đáp ứng đủ bốn tiêu chí ở mục 3.1)

**Kết quả so sánh:**

| Sự kiện | n sự kiện | Tỷ lệ no-repeat 90d | Chênh lệch so với baseline |
|---|---:|---:|---:|
| Baseline (mọi đơn hàng) | 640,828 | 66.48% | — |
| Sau khi trả hàng | 39,552 | 67.84% | **+1.36 điểm %** |
| Sau khi để review thấp (≤ 2 sao) | 14,729 | 68.52% | **+2.04 điểm %** |
| Sau khi mua sản phẩm problematic | 2,316 | 66.71% | +0.23 điểm % *(không đáng kể)* |

**Nhận xét:** Trải nghiệm tiêu cực mạnh (trả hàng, review 1–2 sao) có liên hệ rõ với việc khách không quay lại trong 90 ngày — chênh lệch +1.36pp và +2.04pp là tín hiệu đủ nhất quán. Đây là mối liên hệ thống kê, không phải nhân quả trực tiếp.

Riêng "sau khi mua sản phẩm problematic" cho mức chênh lệch **+0.23pp** — quá gần 0 để kết luận có liên hệ thực sự; con số này đổi dấu tùy cách tính baseline, không nên dùng như tín hiệu churn độc lập. Đây phản ánh việc khách mua nhóm sản phẩm này phần lớn là khách active, vốn đã có tỷ lệ quay lại tương đương mặt bằng chung.

![No-repeat 90d by event](docs/assets/eda_insight_report/figures/churn_no_repeat_90d_by_event.png)

![Churn 180d by exposure](docs/assets/eda_insight_report/figures/churn_180d_by_exposure.png)

---

### 3.6. Phân tích thời gian giao hàng — Sản phẩm rủi ro vs. Toàn platform

**Mục tiêu:** Kiểm định xem lý do `late_delivery` chiếm 20.3% trong nhóm rủi ro (gấp đôi platform) có phản ánh thực trạng giao hàng chậm hay không.

**Hai chỉ số đo lường:**
- `fulfillment_days`: ngày đặt hàng → ngày nhận hàng (tổng trải nghiệm khách)
- `transit_days`: ngày gửi hàng → ngày nhận hàng (leg vận chuyển)

**Kết quả tổng thể (566,067 đơn có dữ liệu shipment):**

| Chỉ số | Platform (all) | Nhóm rủi ro | Không rủi ro |
|---|---:|---:|---:|
| Mean fulfillment | 6.00 ngày | 5.95 ngày | 6.00 ngày |
| Median | 6 | 6 | 6 |
| P90 | 9 | 9 | 9 |
| P99 | 10 | 10 | 10 |
| Mean transit | 4.50 ngày | 4.49 ngày | 4.50 ngày |
| Tỷ lệ đơn "late" (≥ P90 = 9 ngày) | — | 12.55% | 12.47% |

Cả hai chiều đo đều **không có sự khác biệt** giữa nhóm rủi ro và toàn platform.

**Kết quả theo từng SKU rủi ro (fulfillment_days):**

| Sản phẩm | n đơn | Mean | Median | P90 | P99 |
|---|---:|---:|---:|---:|---:|
| SaigonCore YY-50 | 96 | 6.05 | 6 | 9 | 10 |
| HanoiStreet UC-37 | 141 | 5.85 | 6 | 9 | 10 |
| SaigonFlex UC-34 | 228 | 5.96 | 6 | 9 | 10 |
| LotusWear UR-18 | 55 | 5.53 | 5 | 8 | 9.5 |
| MekongFit UC-16 | 90 | 6.07 | 6 | 9 | 10 |
| MekongFit RP-33 | 131 | 6.06 | 6 | 9 | 10 |

Không có SKU nào cho thấy thời gian giao hàng vượt trội so với mặt bằng.

**Nghịch lý "late delivery":**

Khi đào sâu vào 13 đơn trả hàng từ nhóm rủi ro với lý do `late_delivery`, phân phối thực tế là:

| fulfillment_days | Số đơn |
|---:|---:|
| 2 | 1 |
| 4 | 2 |
| 5 | 2 |
| 6 | 3 |
| 7 | 3 |
| 9 | 1 |
| 10 | 1 |

Mean = **6.00 ngày** — bằng hệt mặt bằng platform. Một số đơn giao chỉ **2–4 ngày** vẫn được ghi nhận lý do trả là "giao hàng trễ". Cùng chiều, trên toàn bộ 3,986 đơn trả với lý do `late_delivery`, chỉ **12.6%** thực sự có `fulfillment_days ≥ P90`; mean fulfillment của nhóm này là **5.98 ngày** — không khác gì các lý do trả khác.

**Chẩn đoán:** Tỷ lệ `late_delivery` cao trong nhóm rủi ro **không phản ánh vấn đề logistics**. Đây là hiện tượng *attribution shift* (dịch chuyển quy kết): khách hàng không hài lòng về sản phẩm (sai size, không đúng mô tả) nhưng chọn "giao hàng trễ" như lý do trả hàng dễ chấp nhận hơn về mặt xã hội. Điều này cũng lý giải tại sao lý do `defective` của nhóm rủi ro thấp hơn bình quân đến 12 điểm % — khách không nhận diện vấn đề là lỗi sản phẩm mà quy về nguyên nhân ngoại lai.

**Hàm ý chiến lược:** Hành động cần tập trung vào **kỳ vọng sản phẩm** (size chart, ảnh thực tế, fit guide), không phải carrier SLA. Chi tiết xem Trụ cột 1, mục 4.

![Shipping stats comparison](docs/assets/eda_insight_report/figures/shipping_stats_comparison.png)

![Shipping per SKU risky](docs/assets/eda_insight_report/figures/shipping_per_sku_risky.png)

![Shipping CDF returned only](docs/assets/eda_insight_report/figures/shipping_cdf_returned_only.png)

![Shipping mean by return reason](docs/assets/eda_insight_report/figures/shipping_mean_by_return_reason.png)

---

### 3.7. Liên hệ giữa 814 SKU không có đơn hàng và tồn kho

Kết quả đối soát với `inventory.csv`:
- Tổng SKU unsold: **814**
- SKU unsold có xuất hiện trong inventory: **26 SKU (3.2%)**
- SKU unsold không xuất hiện trong inventory: **788 SKU (96.8%)**

Tại snapshot cuối kỳ `2022-12-31`:
- Chỉ còn **3 SKU unsold** có tồn kho, tổng `stock_on_hand = 9` units.
- Tỷ trọng tồn cuối kỳ của nhóm unsold gần như bằng 0:
  - Theo số lượng tồn: ~**0.01%**
  - Theo giá trị tồn (COGS): ~**0.01%**

Mức độ tồn kéo dài của 26 SKU unsold có xuất hiện trong inventory:
- **23/26 SKU** chỉ xuất hiện **1 tháng snapshot**
- **2 SKU** xuất hiện 2 tháng
- **1 SKU** xuất hiện 3 tháng

**Nhận xét:** Liên hệ giữa nhóm 814 SKU unsold và bài toán hàng tồn kho là **rất yếu**. Đây chủ yếu là vấn đề **catalog hygiene/assortment governance**, không phải tồn kho quy mô lớn.

Điều này trả lời trực tiếp câu hỏi "vì sao 10 năm không bán và cũng không có tồn kho":
- Nếu một SKU không xuất hiện trong `order_items.csv`, ta chỉ biết chắc rằng SKU đó không phát sinh đơn hàng trong giai đoạn 2012-2022.
- Nếu SKU đó cũng không xuất hiện trong `inventory.csv`, khả năng cao SKU chưa từng được nhập kho trong các snapshot quan sát được, hoặc đã không thuộc luồng vận hành tồn kho tại thời điểm chốt tháng.
- Do đó, nhóm này nhiều khả năng là **catalog dư/không còn active/chưa từng launch đầy đủ**, hơn là nhóm hàng vật lý bị tồn đọng.

Tác động business của nhóm này phụ thuộc vào trạng thái listing thực tế:

| Trạng thái thực tế | Mức độ cần quan tâm | Tác động business |
|---|---|---|
| Không active, không hiển thị cho khách, không có tồn kho | Thấp | Gần như không ảnh hưởng doanh thu hay working capital; chủ yếu là nhiễu dữ liệu và chi phí quản trị catalog |
| Vẫn active/hiển thị nhưng không có tồn kho | Trung bình | Gây dead-end trong trải nghiệm tìm kiếm, giảm conversion, làm xấu chỉ số availability |
| Vẫn active, có thể đặt mua nhưng không được fulfillment | Cao | Rủi ro hủy đơn, hoàn tiền, khiếu nại, giảm niềm tin khách hàng |
| Có tồn kho thực tế nhưng không lên snapshot hoặc không được bán | Cao | Rủi ro sai lệch dữ liệu tồn kho, thất thoát cơ hội doanh thu, cần audit vận hành |

Với dữ liệu hiện có, mức độ ảnh hưởng tài chính trực tiếp là rất nhỏ: cuối kỳ chỉ còn **3 SKU** thuộc nhóm unsold có tồn kho, tổng **9 units**, chiếm khoảng **0.01%** cả về số lượng và giá trị tồn. Vì vậy, nhóm này không nên là ưu tiên xử lý tồn kho, nhưng vẫn nên được xử lý như một bài toán chất lượng catalog.

![Unsold inventory coverage](eda_outputs/figures/unsold_inventory_coverage.png)

![Unsold inventory months distribution](eda_outputs/figures/unsold_inventory_months_distribution.png)

---

## 4. Chiến lược kinh doanh đề xuất

Dựa trên toàn bộ insight trên, chúng tôi đề xuất ba trụ cột chiến lược:

---

### Trụ cột 1 — Bảo vệ doanh thu từ nhóm sản phẩm rủi ro

**Vấn đề:** 6 SKU đang tạo ra vòng xoáy tiêu cực: bán nhiều → trả nhiều → review kém → mất khách.

**Hành động đề xuất:**

| Hành động | Ưu tiên | Kỳ vọng |
|---|---|---|
| Kiểm toán size chart và bổ sung fit guide cho các SKU có `wrong_size` cao | Cao | Giảm return rate 20–30% |
| Cập nhật ảnh sản phẩm và mô tả để khớp thực tế: `not_as_described` chiếm ~16% returns của nhóm rủi ro | Cao | Giảm return do kỳ vọng sai |
| Bổ sung tùy chọn lý do trả hàng chi tiết hơn (ví dụ: phân tách "giao trễ so với cam kết" vs "giao trễ do tôi cần gấp") để thu signal thực từ khách | Trung bình | Cải thiện chất lượng dữ liệu return reason, tránh attribution shift |
| Thiết lập workflow tự động can thiệp trong 7 ngày sau sự kiện trả hàng hoặc review thấp (voucher đổi size, hỗ trợ hậu mãi) | Trung bình | Giảm no-repeat 90d ~1–2 điểm % |
| Theo dõi cohort `after_return` và `after_low_rating` theo `repurchase_30/60/90d` | Trung bình | Đo lường hiệu quả can thiệp |

> **Lưu ý:** "Audit SLA vận chuyển" đã bị loại khỏi danh sách ưu tiên. Phân tích thời gian giao hàng thực tế (mục 3.6) xác nhận fulfillment của 6 SKU rủi ro không khác platform (mean 5.95 ngày, P99 10 ngày). Lý do `late_delivery` cao gấp đôi là *attribution shift* của khách hàng không hài lòng về sản phẩm, không phải vấn đề logistics.

---

### Trụ cột 2 — Làm sạch và phân loại 814 SKU chưa phát sinh doanh thu

**Vấn đề:** Hơn 800 SKU chưa tạo doanh thu. Dữ liệu inventory cho thấy phần lớn không nằm trong tồn kho hiện hành, nên đây trước hết là bài toán **catalog hygiene và quản trị assortment**, không phải gánh nặng tồn kho. Chỉ nên kích hoạt bán lại những SKU còn active về mặt chiến lược và có khả năng fulfillment.

**Hành động đề xuất:**

| Phân nhóm | Dấu hiệu nhận biết | Chiến lược |
|---|---|---|
| SKU không active/không có kế hoạch nhập hàng | Không có tồn kho, không có đơn, không có kế hoạch launch | Archive/deactivate để giảm nhiễu catalog và báo cáo |
| SKU active nhưng không có tồn kho | Hiển thị cho khách nhưng `stock_on_hand = 0` | Ẩn khỏi storefront hoặc gắn trạng thái out-of-stock rõ ràng |
| SKU có tiềm năng và có thể fulfillment | Danh mục đang có nhu cầu, giá hợp lý, có kế hoạch nhập hàng | Đẩy listing, tăng visibility, thêm vào flash sale |
| SKU lỗi thời hoặc không phù hợp thị trường | Không có đơn trong thời gian dài, traffic thấp | Tạm ẩn listing, thay đổi content/định vị, chỉ clearance khi thực sự có tồn kho |

**Mục tiêu:** Không phải ép toàn bộ 814 SKU phát sinh doanh thu. Mục tiêu thực tế hơn là phân loại 100% SKU theo trạng thái active/inactive/needs-stock/archive, sau đó chỉ đặt mục tiêu doanh thu cho nhóm còn có lý do kinh doanh để bán.

---

### Trụ cột 3 — Tăng giá trị đơn hàng qua Cross-sell và Bundle

**Vấn đề:** 110 association rules ban đầu bị trùng theo hai chiều; sau khi dedup còn 55 cặp sản phẩm duy nhất. Insight đáng dùng nhất không phải "lift rất cao", mà là các cặp cùng dòng có volume lớn và confidence xấp xỉ 50%.

**Hành động đề xuất theo hai tầng:**

**Tầng 1 — Bundle quy mô lớn (dựa trên support cao):**
- Tạo module "Complete the set" cho top 20 cặp dedup theo `co_orders`
- Ưu tiên HanoiStreet vì 13/20 cặp top-volume thuộc thương hiệu này
- Áp dụng trên PDP, cart, và email/push sau khi khách thêm một SKU nhưng chưa thêm SKU còn lại
- Chạy A/B test với KPI chính là attach rate của SKU còn lại, AOV uplift, và margin sau discount

**Tầng 2 — Recommendation cá nhân hoá (dựa trên lift cao):**
- Chỉ dùng cặp lift cao nhưng volume thấp cho recommendation cá nhân hoá, không dùng làm campaign đại trà
- Cặp như `VietMode RP-75/76` phù hợp hiển thị cho khách đang xem đúng sản phẩm đó, vì quan hệ rất đặc thù nhưng quy mô nhỏ
- KPI chính là CTR và add-to-cart rate của recommendation widget, không phải doanh thu toàn site

**Đo lường hiệu quả:**
- KPI chính: attach rate của sản phẩm ghép cặp, uplift AOV, incremental gross margin, conversion rate của bundle
- Theo dõi sau 30/60/90 ngày triển khai

---


## 5. File kết quả liên quan

- `docs/assets/eda_insight_report/data/products_returned_and_low_rated.csv`
- `docs/assets/eda_insight_report/data/products_not_purchased.csv`
- `docs/assets/eda_insight_report/data/copurchase_pairs_top_by_orders.csv`
- `docs/assets/eda_insight_report/data/copurchase_pairs_top_by_lift.csv`
- `docs/assets/eda_insight_report/data/churn_linkage_summary.csv`
- `docs/assets/eda_insight_report/data/return_reason_overall.csv`
- `docs/assets/eda_insight_report/data/return_reason_problematic_products.csv`
- `docs/assets/eda_insight_report/data/return_reason_comparison_problematic_vs_all.csv`
- `docs/assets/eda_insight_report/data/apriori_frequent_itemsets.csv`
- `docs/assets/eda_insight_report/data/apriori_rules.csv`
- `docs/assets/eda_insight_report/data/apriori_pair_rules_deduped.csv`
- `docs/assets/eda_insight_report/data/apriori_pair_rules_deduped_top20.csv`
- `docs/assets/eda_insight_report/data/apriori_rules_top20_lift.csv`
- `docs/assets/eda_insight_report/data/apriori_rules_top20_support.csv`
- `docs/assets/eda_insight_report/data/apriori_rules_top20_confidence.csv`
- `docs/assets/eda_insight_report/data/inventory_unsold_linkage_by_product.csv`
- `docs/assets/eda_insight_report/data/inventory_ending_snapshot_with_unsold_flag.csv`
- `docs/assets/eda_insight_report/data/inventory_unsold_linkage_summary.csv`
- `docs/assets/eda_insight_report/data/inventory_unsold_ending_summary.csv`
- `docs/assets/eda_insight_report/figures/problematic_products_top15.png`
- `docs/assets/eda_insight_report/figures/unsold_distribution.png`
- `docs/assets/eda_insight_report/figures/copurchase_top15_by_orders.png`
- `docs/assets/eda_insight_report/figures/copurchase_lift_scatter.png`
- `docs/assets/eda_insight_report/figures/churn_no_repeat_90d_by_event.png`
- `docs/assets/eda_insight_report/figures/churn_180d_by_exposure.png`
- `docs/assets/eda_insight_report/figures/return_reason_overall.png`
- `docs/assets/eda_insight_report/figures/return_reason_all_vs_problematic.png`
- `docs/assets/eda_insight_report/figures/apriori_dedup_top15_volume.png`
- `docs/assets/eda_insight_report/figures/apriori_dedup_lift_vs_volume.png`
- `docs/assets/eda_insight_report/figures/unsold_inventory_coverage.png`
- `docs/assets/eda_insight_report/figures/unsold_inventory_months_distribution.png`
- `docs/assets/eda_insight_report/data/shipping_fulfillment_stats.csv`
- `docs/assets/eda_insight_report/data/shipping_transit_stats.csv`
- `docs/assets/eda_insight_report/data/shipping_per_sku_risky.csv`
- `docs/assets/eda_insight_report/data/shipping_returned_orders_stats.csv`
- `docs/assets/eda_insight_report/figures/shipping_stats_comparison.png`
- `docs/assets/eda_insight_report/figures/shipping_per_sku_risky.png`
- `docs/assets/eda_insight_report/figures/shipping_cdf_risky_vs_nonrisky.png`
- `docs/assets/eda_insight_report/figures/shipping_cdf_returned_only.png`
- `docs/assets/eda_insight_report/figures/shipping_mean_by_return_reason.png`
