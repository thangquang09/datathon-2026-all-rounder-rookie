# Phase 2 Business Insights

Nguồn: `group_job/phase2_product_insights_eda.ipynb` và các bảng xuất trong `group_job/tables`.

## Executive Summary

Phase 2 cho thấy bài toán kinh doanh chính không phải là “bán thêm sản phẩm đang bán chạy”, mà là **bảo vệ profit pool, kiểm soát promotion, giảm leakage từ return và dùng inventory đúng thời điểm**.

Tổng revenue quan sát được là khoảng **16.43B**, gross margin khoảng **2.27B**. Tuy nhiên profit không phân bổ đều: top 10 sản phẩm theo gross margin tạo **23.5%** tổng gross margin, trong khi top 10 sản phẩm theo revenue chỉ tạo **15.6%** tổng revenue. Top 12 theo số lượng bán và top 12 theo gross margin chỉ trùng **5 sản phẩm**, nên “best seller” không tự động là “best product to scale”.

## 1. Profit Pool: Streetwear là core, nhưng cần quản trị theo SKU

`Streetwear` là category lớn nhất, đóng góp khoảng **79.9% revenue** và **1.74B gross margin**. Đây là profit pool chính của business. Tuy nhiên Streetwear cũng chứa cả SKU rất mạnh và SKU lỗ nặng.

Các product đáng scale theo margin gồm:

| Product | Category | Units | Revenue | Gross margin | Margin rate |
|---|---:|---:|---:|---:|---:|
| SaigonFlex UM-43 | Streetwear | 31,471 | 340.71M | 130.46M | 38.3% |
| UrbanVN UE-05 | Streetwear | 35,844 | 185.08M | 77.68M | 42.0% |
| SaigonFlex UC-69 | Streetwear | 36,515 | 207.93M | 56.69M | 27.3% |
| SaigonFlex UM-48 | Streetwear | 12,379 | 123.95M | 47.50M | 38.3% |

Business implication: nên ưu tiên stock, campaign và visibility cho nhóm SKU có **gross margin cao + demand cao**, thay vì chỉ nhìn volume. Một số SKU volume lớn nhưng margin thấp hoặc âm cần được đưa vào nhóm kiểm soát riêng.

## 2. Product Loss: bán lỗ không hoàn toàn đồng nghĩa xả tồn

Có **359 sản phẩm gross margin âm**, chiếm khoảng **30.5% revenue** nhưng làm mất khoảng **122.36M gross margin**. Promo revenue share của nhóm này là **33.8%**, gần mức toàn danh mục **33.1%**, nên bản thân promotion không giải thích hết việc lỗ.

Các sản phẩm lỗ lớn nhất:

| Product | Category | Units | Revenue | Gross margin | Promo share | Inventory signal |
|---|---:|---:|---:|---:|---:|---|
| HanoiStreet UM-10 | Streetwear | 28,993 | 342.26M | -8.37M | 32.4% | overstock rate 99.2%, sell-through 5.9% |
| HanoiStreet UE-36 | Streetwear | 31,220 | 168.30M | -5.51M | 33.4% | overstock rate 94.9%, sell-through 7.8% |
| SaigonFlex UM-96 | Streetwear | 24,485 | 251.82M | -5.45M | 31.9% | days supply rất cao |

Insight quan trọng: dữ liệu cho phép tách SKU lỗ thành các nhóm rõ hơn, thay vì kết luận chung “bán lỗ = xả hàng”.

| Nhóm SKU lỗ | Bằng chứng trong dữ liệu | Ví dụ SKU | Business action |
|---|---|---|---|
| **Overstock clearance** | 127/359 SKU lỗ có `overstock_rate >= 70%` và `sell-through <= 15%`. Nhóm này chiếm **67.2% revenue của toàn bộ SKU lỗ** và khoảng **76.91M gross margin âm**. | HanoiStreet UM-10, HanoiStreet UE-36, SaigonFlex UM-96 | Có cơ sở dùng markdown/promotion để giải phóng tồn, nhưng cần giới hạn thời gian và không scale thêm stock. |
| **Loss SKU có downstream margin dương** | Một số SKU lỗ có khách quay lại mua trong 90 ngày và tạo gross margin sau đó lớn hơn phần lỗ ban đầu. Ví dụ HanoiStreet UE-36 có **6,134 buyers**, repeat 90 ngày **34.4%**, downstream margin **9.40M** so với initial loss **5.02M**. | HanoiStreet UE-36, MekongFit UE-18, HanoiStreet UM-10, UrbanVN UE-14 | Có thể xem là acquisition candidate, nhưng chỉ là tín hiệu association. Cần holdout/A-B test trước khi chủ động bán lỗ để acquire khách. |
| **Pricing/COGS issue** | 232 SKU lỗ không nằm trong nhóm clearance rõ ràng nhưng realized price/unit thấp hơn COGS. Nhóm này chiếm **32.8% revenue của SKU lỗ**. | SaigonFlex UM-12, SaigonFlex UC-16, SaigonFlex UC-64 | Cần sửa discount cap, price floor hoặc COGS. Nếu không có downstream margin đủ bù lỗ thì nên ngừng scale. |

Vì vậy phần “bán lỗ” nên được quản trị bằng rule: clearance nếu tồn cao, acquisition only nếu có cohort evidence, còn lại phải sửa pricing/COGS. Không nên gom tất cả SKU lỗ vào một chiến lược promotion chung.

## 3. Seasonality: cần lên kế hoạch inventory và promotion theo tháng peak

Seasonality đủ rõ để dùng cho planning:

| Category | Peak month | Seasonality index |
|---|---:|---:|
| GenZ | Tháng 6 | 1.75x |
| Streetwear | Tháng 5 | 1.55x |
| Casual | Tháng 5 | 1.51x |
| Outdoor | Tháng 12 | 1.38x |

Business implication: không nên chạy một lịch promotion chung cho mọi category. Streetwear/Casual nên được chuẩn bị stock và campaign trước tháng 5, GenZ trước tháng 6, Outdoor trước tháng 12.

## 4. Promotion: là đòn bẩy lớn nhưng cần guardrail margin

Promotion đóng góp khoảng **33.1% revenue**, nhưng hiệu quả rất khác nhau theo channel.

| Promo channel | Revenue | Gross margin | Margin rate | Profit / customer |
|---|---:|---:|---:|---:|
| email | 900.63M | 92.67M | 10.3% | 3.66K |
| all_channels | 2.34B | 127.22M | 5.4% | 2.68K |
| social_media | 689.17M | 18.79M | 2.7% | 861 |
| online | 1.42B | -173.15M | -12.2% | -4.83K |

`online` promo là điểm đỏ: revenue lớn nhưng gross margin âm sâu. Không nên scale online promotion nếu chưa khóa guardrail về margin, SKU eligibility và discount cap.

Email có chất lượng tốt hơn: trong nhóm khách đã mua bằng email promo, cứ 10 khách thì khoảng **3.0 khách** có đơn tiếp theo trong 90 ngày, tạo trung bình khoảng **15.73K gross margin** sau đó. Tuy nhiên đây là association, không phải causal impact, vì dataset không có nhóm khách được email target nhưng không mua. Kết luận đúng nên là: **email-promo purchasers có downstream profit tốt**, còn để đo “email target 10 người tạo bao nhiêu profit” cần A/B test hoặc holdout.

## 5. Outdoor và HanoiStreet: demand có thật, vấn đề nằm ở tồn kho SKU

Outdoor không phải category yếu. Outdoor tạo khoảng **2.49B revenue**, **408.52M gross margin**, margin rate **16.4%**, cao hơn Streetwear margin rate **13.2%**.

Trong Outdoor, `HanoiStreet` là brand lớn nhất:

| Brand | Units | Revenue | Gross margin | Margin rate | Promo share | Avg days supply | Sell-through | Overstock rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HanoiStreet | 530,071 | 1.04B | 216.19M | 20.7% | 37.5% | 1,921 | 12.4% | 78.4% |
| VietMotion | 305,380 | 596.77M | 76.52M | 12.8% | 36.0% | 1,500 | 11.6% | 80.4% |
| VietMode | 98,283 | 355.39M | 57.76M | 16.3% | 34.1% | 208 | 22.7% | 45.5% |

Insight: không nên nói “HanoiStreet bán không tốt” ở cấp brand. HanoiStreet Outdoor vẫn có demand và margin tốt. Vấn đề là **stock quality**: days of supply và overstock rate quá cao, tức có nhiều SKU bị tồn lâu. Cách xử lý nên là SKU-level markdown/clearance, không phải cắt toàn bộ brand.

## 6. Return Leakage: wrong size là vấn đề lớn nhất

Return reason lớn nhất là:

| Reason | Share |
|---|---:|
| wrong_size | 35.0% |
| defective | 20.1% |
| not_as_described | 17.6% |
| changed_mind | 17.4% |
| late_delivery | 10.0% |

Điểm review trung bình không đủ để giải thích return hoặc margin âm, vì rating của nhóm sản phẩm lỗ gần tương đương nhóm không lỗ. Do đó không nên dùng rating trung bình làm proxy cho product quality.

Business action:

1. Với `wrong_size`: cải thiện size chart, fit guide, ảnh/model sizing, hoặc warning theo SKU có return cao.
2. Với `defective`: audit supplier/QC theo SKU và batch.
3. Với `not_as_described`: sửa mô tả sản phẩm, ảnh, material/spec copy.
4. Với SKU return cao nhưng margin tốt: ưu tiên sửa leakage, không vội dừng bán.

## 7. Inventory: có hai bài toán khác nhau

Notebook chỉ ra hai vấn đề inventory song song:

1. **Protect peak demand:** sản phẩm margin cao hoặc demand cao cần đủ hàng trước peak month.
2. **Reduce overstock:** một số SKU có stock lớn, days supply cao, sell-through thấp.

Top overstock candidates:

| Product | Category | Revenue | Gross margin | Days supply | Sell-through | Overstock rate |
|---|---:|---:|---:|---:|---:|---:|
| HanoiStreet RP-79 | Outdoor | 34.00M | 7.10M | 2,024.5 | 10.1% | 88.3% |
| HanoiStreet RP-80 | Outdoor | 33.82M | 4.13M | 1,979.3 | 9.7% | 88.3% |
| SaigonFlex UC-69 | Streetwear | 207.93M | 56.69M | 2,493.1 | 6.6% | 97.6% |
| UrbanVN UE-05 | Streetwear | 185.08M | 77.68M | 2,250.3 | 8.0% | 92.8% |

Điểm đáng chú ý: một số sản phẩm vừa margin cao vừa overstock cao. Với nhóm này không nên markdown quá mạnh ngay; nên kiểm tra seasonality và stock planning trước, vì có thể hàng tồn cao do chuẩn bị cho peak demand.

## 8. Regression Model: dùng để ưu tiên replenishment, không dùng để kết luận causal promotion

Notebook thử model `HistGradientBoostingRegressor` dự báo `units_sold` tháng tới theo product-month. Kết quả holdout từ 2022 trở đi:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| HistGradientBoostingRegressor | 3.63 | 6.61 | 0.872 |
| Naive current-month units sold | 4.28 | 8.40 | 0.794 |

Model này hữu ích cho bài toán **stock-gap prioritization**: SKU nào dự báo bán tháng tới cao hơn stock hiện có thì cần ưu tiên replenishment. Ví dụ các SKU có gap dự báo cao gồm `HanoiStreet RP-08`, `HanoiStreet RP-07`, `MekongFit UE-13`.

Không nên dùng model inventory này để nói promotion gây tăng/giảm demand, vì inventory table không có exposure promotion theo tháng. Muốn đo promo uplift cần nối order-level campaign hoặc thiết kế holdout.

## Recommended Business Actions

### A. Scale có chọn lọc

Scale các SKU có gross margin cao, demand cao, return thấp, ví dụ nhóm `SaigonFlex UM-43`, `UrbanVN UE-05`, `SaigonFlex UC-69`. Không scale chỉ vì volume cao.

### B. Tạo promotion guardrail

Thiết lập rule trước khi chạy promotion:

- Không cho SKU gross margin âm vào campaign acquisition nếu không có repeat-profit evidence.
- Với online promo, cần discount cap và SKU whitelist vì channel này đang âm margin.
- Email nên được test bằng holdout để đo incremental lift thật.

### C. Xử lý SKU lỗ theo nguyên nhân

Không gom chung toàn bộ sản phẩm lỗ. Tách thành:

- Lỗ do tồn kho: clearance có giới hạn thời gian.
- Lỗ do return/quality: sửa product content, size, QC.
- Lỗ do pricing/COGS: điều chỉnh giá hoặc ngừng scale.
- Lỗ có repeat profit: giữ như acquisition SKU nhưng phải đo cohort.

### D. Chỉnh inventory theo seasonality

Lập lịch replenishment trước peak:

- Streetwear/Casual: trước tháng 5.
- GenZ: trước tháng 6.
- Outdoor: trước tháng 12.

Với SKU overstock nhưng có margin tốt, không giảm giá đại trà trước khi kiểm tra peak month.

### E. Giảm return leakage

Ưu tiên xử lý `wrong_size`, vì chiếm khoảng 35% return. Đây là action có khả năng cải thiện profit mà không cần tăng traffic hay tăng discount.

## Caveats

- Dataset không có cost của lưu kho theo SKU, nên chưa tính được inventory holding cost thật.
- Dataset không có danh sách khách được target nhưng không mua, nên promotion analysis là association, không phải causal uplift.
- Review title khá generic, không đủ sâu để làm text mining nguyên nhân sản phẩm lỗ.
- Inventory không có warehouse/region dimension, nên demand theo vùng chưa thể chuyển thẳng thành quyết định đặt kho vùng.

## One-line Narrative For Report

Business đang có profit pool rõ ở Streetwear và một số SKU margin cao, nhưng lợi nhuận bị kéo xuống bởi promotion thiếu guardrail, SKU lỗ/tồn kho, và return do sizing/quality; chiến lược tốt nhất là scale SKU có margin thật, dùng promotion có điều kiện, xử lý return leakage, và lập inventory theo seasonality thay vì theo volume tổng.
