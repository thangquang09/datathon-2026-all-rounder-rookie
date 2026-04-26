# Phase 2 Business Insights

Nguồn: `group_job/phase2_product_insights_eda.ipynb` và các bảng xuất trong `group_job/tables`.
Tất cả metric đã được audit độc lập trong `group_job/phase2_metric_audit.md` — 36/36 checks PASS.

## Executive Summary

Bài toán chính không phải là "bán thêm sản phẩm đang bán chạy", mà là **bảo vệ profit pool, kiểm soát promotion, giảm return leakage, và dùng inventory đúng thời điểm**.

| Chỉ số | Giá trị |
|---|---:|
| Tổng revenue | **~16.43B** |
| Tổng gross margin | **~2.27B** |
| Top-10 product gross margin share | **23.5%** tổng GM |
| Top-10 product revenue share | **15.6%** tổng revenue |
| Top-12 units ∩ Top-12 margin | **5/12 sản phẩm** |

Profit và volume là hai thứ khác nhau: "best seller" chưa chắc là sản phẩm nên scale.

## 1. Profit Pool: Streetwear là core, nhưng cần quản trị theo SKU

`Streetwear` đóng góp **~79.9% revenue** và **1.74B gross margin** — đây là profit pool chính. Tuy nhiên Streetwear chứa cả SKU rất mạnh lẫn SKU lỗ nặng, nên không thể quản lý ở cấp category.

**Sản phẩm đáng scale** (margin cao + demand cao):

| Product | Category | Units | Revenue | Gross margin | Margin rate |
|---|---:|---:|---:|---:|---:|
| SaigonFlex UM-43 | Streetwear | 31,471 | 340.71M | 130.46M | 38.3% |
| UrbanVN UE-05 | Streetwear | 35,844 | 185.08M | 77.68M | 42.0% |
| SaigonFlex UC-69 | Streetwear | 36,515 | 207.93M | 56.69M | 27.3% |
| SaigonFlex UM-48 | Streetwear | 12,379 | 123.95M | 47.50M | 38.3% |

**Action:** Ưu tiên stock, campaign, visibility cho SKU có gross margin cao + demand cao. SKU volume lớn nhưng margin thấp/âm cần đưa vào nhóm kiểm soát riêng.

## 2. Product Loss: bán lỗ không đồng nghĩa xả tồn — cần phân nhóm

Có **359 sản phẩm gross margin âm**, chiếm **~30.5% revenue** nhưng gây mất **~122.36M gross margin**. Promo revenue share nhóm lỗ là **33.8%** (gần mức toàn danh mục 33.1%), nên promotion không giải thích hết việc lỗ.

**Sản phẩm lỗ lớn nhất:**

| Product | Category | Units | Revenue | Gross margin | Promo share | Inventory signal |
|---|---:|---:|---:|---:|---:|---|
| HanoiStreet UM-10 | Streetwear | 28,993 | 342.26M | -8.37M | 32.4% | overstock 99.2%, sell-through 5.9% |
| HanoiStreet UE-36 | Streetwear | 31,220 | 168.30M | -5.51M | 33.4% | overstock 94.9%, sell-through 7.8% |
| SaigonFlex UM-96 | Streetwear | 24,485 | 251.82M | -5.45M | 31.9% | days supply rất cao |

Thay vì gom tất cả SKU lỗ vào một chiến lược, cần phân thành 3 nhóm:

| Nhóm SKU lỗ | Bằng chứng | Ví dụ SKU | Action |
|---|---|---|---|
| **Overstock clearance** (127/359 SKU) | overstock ≥ 70%, sell-through ≤ 15%, chiếm **67.2% revenue nhóm lỗ** và **76.91M GM âm** | HanoiStreet UM-10, UE-36, SaigonFlex UM-96 | Markdown/clearance có thời hạn. Không scale thêm stock. |
| **Loss SKU có downstream margin dương** | Khách mua SKU lỗ quay lại trong 90 ngày, tạo GM sau đó > initial loss. VD: HanoiStreet UE-36 có **6,134 buyers**, repeat 90d **34.4%**, downstream margin **9.40M** > loss **5.02M** | UE-36, MekongFit UE-18, UM-10, UrbanVN UE-14 | Xem là acquisition candidate — nhưng đây là association, không phải causal. Cần holdout/A-B test trước khi chủ động bán lỗ để acquire khách. |
| **Pricing/COGS issue** (232 SKU) | Không phải clearance rõ ràng, nhưng realized price/unit < COGS. Chiếm **32.8% revenue nhóm lỗ** | SaigonFlex UM-12, UC-16, UC-64 | Sửa discount cap, price floor hoặc COGS. Nếu downstream margin không đủ bù → ngừng scale. |

**Lưu ý:** SKU lỗ cũng có seasonality riêng — cần kiểm tra peak month trước khi clearance. Nếu SKU lỗ có demand spike theo mùa, clearance sai thời điểm sẽ bỏ lỡ cơ hội bán.

## 3. Seasonality: lên kế hoạch inventory và promotion theo tháng peak

Seasonality đủ rõ để dùng cho planning:

| Category | Peak month | Seasonality index |
|---|---:|---:|
| GenZ | Tháng 6 | **1.75x** |
| Streetwear | Tháng 5 | **1.55x** |
| Casual | Tháng 5 | **1.51x** |
| Outdoor | Tháng 12 | **1.38x** |

**Action:** Mỗi category cần lịch stock và campaign riêng. Chuẩn bị trước peak 1-2 tháng:
- Streetwear/Casual: sẵn sàng trước **tháng 4**
- GenZ: sẵn sàng trước **tháng 5**
- Outdoor: sẵn sàng trước **tháng 11**

## 4. Promotion: đòn bẩy lớn nhưng cần guardrail margin

Promotion đóng góp **~33.1% revenue**, nhưng hiệu quả rất khác nhau theo channel.

| Promo channel | Revenue | Gross margin | Margin rate | Profit/customer |
|---|---:|---:|---:|---:|
| email | 900.63M | 92.67M | **10.3%** | 3.66K |
| all_channels | 2.34B | 127.22M | 5.4% | 2.68K |
| social_media | 689.17M | 18.79M | 2.7% | 861 |
| online | 1.42B | **-173.15M** | **-12.2%** | -4.83K |

**Điểm đỏ:** `online` promo có revenue lớn nhưng gross margin âm sâu. Không scale online promotion nếu chưa thiết lập guardrail về margin, SKU eligibility, discount cap.

**Email là channel tốt nhất:** Trong nhóm khách mua qua email promo, ~3/10 có đơn tiếp theo trong 90 ngày, tạo trung bình **~15.73K gross margin** sau đó. Tuy nhiên đây là association (dataset không có nhóm control), nên kết luận đúng là: **email-promo purchasers có downstream profit tốt**, còn để đo incremental lift thật cần A/B test.

## 5. Outdoor & HanoiStreet: demand có thật, vấn đề nằm ở tồn kho SKU

Outdoor không phải category yếu:

| Chỉ số | Giá trị |
|---|---:|
| Revenue | **2.49B** |
| Gross margin | **408.52M** |
| Margin rate | **16.4%** (cao hơn Streetwear 13.2%) |

Trong Outdoor, `HanoiStreet` là brand lớn nhất:

| Brand | Units | Revenue | GM | Margin rate | Days supply | Sell-through | Overstock rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HanoiStreet | 530K | 1.04B | 216.19M | 20.7% | 1,921 | 12.4% | 78.4% |
| VietMotion | 305K | 596.77M | 76.52M | 12.8% | 1,500 | 11.6% | 80.4% |
| VietMode | 98K | 355.39M | 57.76M | 16.3% | 208 | 22.7% | 45.5% |

**Insight:** HanoiStreet Outdoor có demand và margin tốt ở cấp brand. Vấn đề là **stock quality**: days of supply và overstock rate quá cao. Cách xử lý đúng là SKU-level markdown/clearance, không phải cắt toàn bộ brand.

**So sánh quan trọng:** VietMode có days supply chỉ 208 và sell-through 22.7% — hiệu quả vận hành tốt hơn hẳn. HanoiStreet và VietMotion cần học từ VietMode về quản lý stock depth.

## 6. Return Leakage: wrong_size là vấn đề lớn nhất

Phân bổ return reason:

| Reason | Share |
|---|---:|
| wrong_size | **35.0%** |
| defective | 20.1% |
| not_as_described | 17.6% |
| changed_mind | 17.4% |
| late_delivery | 10.0% |

**Lưu ý:** Rating trung bình không phân biệt được sản phẩm lỗ/không lỗ, nên không nên dùng rating làm proxy cho product quality.

**Action theo nguyên nhân:**
1. **wrong_size (35%):** Cải thiện size chart, fit guide, ảnh model/sizing. Thêm warning cho SKU có return rate cao. Đây là action cải thiện profit mà không cần tăng traffic hay discount.
2. **defective (20%):** Audit supplier/QC theo SKU và batch.
3. **not_as_described (18%):** Sửa mô tả sản phẩm, ảnh, material/spec.
4. **SKU return cao nhưng margin tốt:** Ưu tiên sửa leakage, không vội dừng bán.

## 7. Geography: demand tập trung nhưng chưa đủ cho kho vùng

Demand phân bố không đều theo region:

| Region | Insight |
|---|---|
| Đông Nam Bộ | Demand lớn nhất, chiếm phần lớn revenue |
| Đồng bằng sông Hồng | Demand lớn thứ hai |
| Các vùng khác | Demand thấp hơn đáng kể |

**Action:** Có thể dùng region demand để guide campaign targeting. Tuy nhiên inventory không có dimension warehouse/region, nên chưa thể chuyển trực tiếp thành quyết định đặt kho vùng.

## 8. Inventory: hai bài toán song song

### A. Bảo vệ peak demand
Sản phẩm margin cao/demand cao cần đủ hàng trước peak month. Sử dụng regression model (HistGradientBoostingRegressor, holdout từ 2022+, R²=0.872) để dự báo `units_sold` tháng tới và xác định **stock-gap** — SKU nào dự báo bán cao hơn stock hiện có thì ưu tiên replenishment.

SKU có stock-gap dự báo cao: `HanoiStreet RP-08`, `HanoiStreet RP-07`, `MekongFit UE-13`.

### B. Giảm overstock

Top overstock candidates:

| Product | Category | Revenue | GM | Days supply | Sell-through | Overstock rate |
|---|---:|---:|---:|---:|---:|---:|
| HanoiStreet RP-79 | Outdoor | 34.00M | 7.10M | 2,024.5 | 10.1% | 88.3% |
| HanoiStreet RP-80 | Outdoor | 33.82M | 4.13M | 1,979.3 | 9.7% | 88.3% |
| SaigonFlex UC-69 | Streetwear | 207.93M | 56.69M | 2,493.1 | 6.6% | 97.6% |
| UrbanVN UE-05 | Streetwear | 185.08M | 77.68M | 2,250.3 | 8.0% | 92.8% |

**Điểm cần lưu ý:** Một số SKU vừa margin cao vừa overstock cao (VD: SaigonFlex UC-69, UrbanVN UE-05). Với nhóm này **không markdown quá mạnh ngay** — cần kiểm tra seasonality trước, vì có thể hàng tồn cao do chuẩn bị cho peak demand.

**Lưu ý về model:** Inventory regression model dùng để ưu tiên replenishment, không dùng để kết luận causal effect của promotion (inventory table không có exposure promotion theo tháng). Muốn đo promo uplift cần order-level campaign data hoặc holdout.

---

## Recommended Business Actions

### A. Scale có chọn lọc
Scale SKU có gross margin cao + demand cao + return thấp: `SaigonFlex UM-43`, `UrbanVN UE-05`, `SaigonFlex UC-69`. Không scale chỉ vì volume.

### B. Tạo promotion guardrail
- SKU gross margin âm → không cho vào campaign acquisition nếu không có repeat-profit evidence.
- Online promo → cần discount cap + SKU whitelist (đang âm margin).
- Email → test holdout để đo incremental lift thật.

### C. Xử lý SKU lỗ theo nguyên nhân
- Lỗ do tồn kho: clearance có thời hạn.
- Lỗ do return/quality: sửa product content, size, QC.
- Lỗ do pricing/COGS: điều chỉnh giá hoặc ngừng scale.
- Lỗ có repeat profit: giữ như acquisition SKU nhưng phải đo cohort.

### D. Chỉnh inventory theo seasonality
- Streetwear/Casual: replenishment trước **tháng 5**.
- GenZ: trước **tháng 6**.
- Outdoor: trước **tháng 12**.
- SKU overstock + margin tốt: không markdown đại trà trước khi kiểm tra peak month.

### E. Giảm return leakage
Ưu tiên xử lý `wrong_size` (35% return). Đây là action cải thiện profit mà không cần tăng traffic hay discount.

---

## Caveats

1. Dataset **không có inventory holding cost** theo SKU → chưa tính được chi phí lưu kho thật.
2. Dataset **không có nhóm control** cho promotion → promotion analysis là association, không phải causal uplift.
3. Review title khá generic → không đủ sâu để text mining nguyên nhân sản phẩm lỗ.
4. Inventory **không có warehouse/region dimension** → demand theo vùng chưa chuyển thành quyết định đặt kho.
5. Regression model train trên product-month aggregation, holdout từ 2022 trở đi, 16 features → useful cho prioritization nhưng không phải demand forecast chính xác.

## One-line Narrative

Business có profit pool rõ ở Streetwear và một số SKU margin cao, nhưng lợi nhuận bị kéo xuống bởi promotion thiếu guardrail, SKU lỗ/tồn kho, và return do sizing/quality. Chiến lược đúng: **scale SKU có margin thật, dùng promotion có điều kiện, xử lý return leakage, lập inventory theo seasonality** — thay vì theo volume tổng.
