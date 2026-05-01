# Dataset Description

## Tổng quan Bộ Dữ liệu

Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam, giai đoạn **04/07/2012 – 31/12/2022**, gồm **15 file CSV** chia thành **4 lớp**.

---

## 🗂️ Master — Dữ liệu tham chiếu

### products.csv — Danh mục sản phẩm

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `product_id` | int | Khoá chính |
| `product_name` | str | Tên sản phẩm |
| `category` | str | Danh mục sản phẩm |
| `segment` | str | Phân khúc thị trường |
| `size` | str | Kích cỡ (S/M/L/XL) |
| `color` | str | Nhãn màu sản phẩm |
| `price` | float | Giá bán lẻ |
| `cogs` | float | Giá vốn hàng bán (< price) |


#### Ví dụ dữ liệu thật (products.csv)

| product_id | product_name | category | segment | size | color | price | cogs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | SaigonFlex UR-01 | Streetwear | Standard | S | orange | 2895.329458972047 | 2518.9366293056805 |
| 1027 | MekongStyle UM-01 | Streetwear | Balanced | XL | silver | 14736.2840625 | 13999.469859375 |
| 1176 | MekongFit UE-12 | Streetwear | Performance | S | green | 7474.808502673796 | 7035.289762716578 |

### customers.csv — Khách hàng

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `customer_id` | int | Khoá chính |
| `zip` | int | Mã bưu chính |
| `city` | str | Thành phố |
| `signup_date` | date | Ngày đăng ký |
| `gender` | str | Giới tính (nullable) |
| `age_group` | str | Nhóm tuổi (nullable) |
| `acquisition_channel` | str | Kênh tiếp thị (nullable) |


#### Ví dụ dữ liệu thật (customers.csv)

| customer_id | zip | city | signup_date | gender | age_group | acquisition_channel |
| --- | --- | --- | --- | --- | --- | --- |
| 44591 | 21703 | Bac Giang | 2015-07-15 | Female | 18-24 | email_campaign |
| 70227 | 42241 | Ha Long | 2016-01-31 | Female | 35-44 | social_media |
| 83667 | 64632 | Tam Ky | 2019-12-11 | Female | 18-24 | organic_search |

### promotions.csv — Chương trình khuyến mãi

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `promo_id` | str | Khoá chính |
| `promo_name` | str | Tên chiến dịch |
| `promo_type` | str | percentage hoặc fixed |
| `discount_value` | float | Giá trị giảm |
| `start_date` | date | Ngày bắt đầu |
| `end_date` | date | Ngày kết thúc |
| `applicable_category` | str | Danh mục áp dụng (null = tất cả) |
| `promo_channel` | str | Kênh phân phối (nullable) |
| `stackable_flag` | int | Cho phép áp dụng nhiều KM cùng lúc |
| `min_order_value` | float | Giá trị đơn tối thiểu (nullable) |


#### Ví dụ dữ liệu thật (promotions.csv)

| promo_id | promo_name | promo_type | discount_value | start_date | end_date | applicable_category | promo_channel | stackable_flag | min_order_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PROMO-0024 | Year-End Sale 2017 | percentage | 20.0 | 2017-11-18 | 2018-01-02 |  | all_channels | 0 | 0 |
| PROMO-0038 | Mid-Year Sale 2020 | percentage | 18.0 | 2020-06-23 | 2020-07-22 |  | social_media | 0 | 0 |
| PROMO-0034 | Year-End Sale 2019 | percentage | 20.0 | 2019-11-18 | 2020-01-02 |  | all_channels | 0 | 50000 |

### geography.csv — Địa lý

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `zip` | int | Khoá chính |
| `city` | str | Thành phố |
| `region` | str | Vùng địa lý |
| `district` | str | Quận/huyện |


#### Ví dụ dữ liệu thật (geography.csv)

| zip | city | region | district |
| --- | --- | --- | --- |
| 20784 | Viet Tri | East | District #05 |
| 54493 | Quang Ngai | Central | District #26 |
| 15474 | Bac Ninh | East | District #13 |

---

## 🔄 Transaction — Giao dịch

### orders.csv — Đơn hàng

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | Khoá chính |
| `order_date` | date | Ngày đặt hàng |
| `customer_id` | int | FK → customers |
| `zip` | int | Mã bưu chính giao hàng |
| `order_status` | str | Trạng thái đơn hàng |
| `payment_method` | str | Phương thức thanh toán |
| `device_type` | str | Thiết bị đặt hàng |
| `order_source` | str | Kênh marketing |


#### Ví dụ dữ liệu thật (orders.csv)

| order_id | order_date | customer_id | zip | order_status | payment_method | device_type | order_source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 202620 | 2014-07-17 | 83248 | 64080 | delivered | credit_card | mobile | organic_search |
| 417604 | 2016-07-27 | 98877 | 39218 | delivered | credit_card | mobile | referral |
| 265822 | 2015-04-01 | 152845 | 98801 | delivered | credit_card | mobile | social_media |

### order_items.csv — Chi tiết đơn hàng

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | FK → orders |
| `product_id` | int | FK → products |
| `quantity` | int | Số lượng |
| `unit_price` | float | Đơn giá sau khuyến mãi |
| `discount_amount` | float | Tổng tiền giảm |
| `promo_id` | str | FK → promotions (nullable) |
| `promo_id_2` | str | Khuyến mãi thứ hai (nullable) |


#### Ví dụ dữ liệu thật (order_items.csv)

| order_id | product_id | quantity | unit_price | discount_amount | promo_id | promo_id_2 |
| --- | --- | --- | --- | --- | --- | --- |
| 566340 | 733 | 2 | 4407.73 | 0.0 |  |  |
| 383271 | 792 | 2 | 803.61 | 0.0 |  |  |
| 348583 | 793 | 7 | 775.59 | 1085.83 | PROMO-0014 |  |

### payments.csv — Thanh toán (quan hệ 1:1 với orders)

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | FK → orders |
| `payment_method` | str | Phương thức thanh toán |
| `payment_value` | float | Tổng giá trị thanh toán |
| `installments` | int | Số kỳ trả góp |


#### Ví dụ dữ liệu thật (payments.csv)

| order_id | payment_method | payment_value | installments |
| --- | --- | --- | --- |
| 690160 | paypal | 78087.42 | 12 |
| 195904 | cod | 7321.02 | 1 |
| 703881 | credit_card | 23755.08 | 3 |

### shipments.csv — Vận chuyển (chỉ cho đơn shipped/delivered/returned)

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | FK → orders |
| `ship_date` | date | Ngày gửi hàng |
| `delivery_date` | date | Ngày giao hàng |
| `shipping_fee` | float | Phí vận chuyển |


#### Ví dụ dữ liệu thật (shipments.csv)

| order_id | ship_date | delivery_date | shipping_fee |
| --- | --- | --- | --- |
| 318145 | 2015-08-28 | 2015-09-04 | 0.81 |
| 748694 | 2021-03-13 | 2021-03-15 | 1.94 |
| 104202 | 2013-08-04 | 2013-08-11 | 0.61 |

### returns.csv — Trả hàng

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `return_id` | str | Khoá chính |
| `order_id` | int | FK → orders |
| `product_id` | int | FK → products |
| `return_date` | date | Ngày trả hàng |
| `return_reason` | str | Lý do trả hàng |
| `return_quantity` | int | Số lượng trả |
| `refund_amount` | float | Số tiền hoàn lại |


#### Ví dụ dữ liệu thật (returns.csv)

| return_id | order_id | product_id | return_date | return_reason | return_quantity | refund_amount |
| --- | --- | --- | --- | --- | --- | --- |
| RET-001525 | 23809 | 785 | 2012-10-28 | not_as_described | 1 | 584.12 |
| RET-036875 | 586936 | 1995 | 2018-06-04 | wrong_size | 6 | 29135.86 |
| RET-048659 | 785482 | 2395 | 2022-01-05 | defective | 3 | 4512.95 |

### reviews.csv — Đánh giá sản phẩm

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `review_id` | str | Khoá chính |
| `order_id` | int | FK → orders |
| `product_id` | int | FK → products |
| `customer_id` | int | FK → customers |
| `review_date` | date | Ngày đánh giá |
| `rating` | int | Điểm từ 1–5 |
| `review_title` | str | Tiêu đề đánh giá |


#### Ví dụ dữ liệu thật (reviews.csv)

| review_id | order_id | product_id | customer_id | review_date | rating | review_title |
| --- | --- | --- | --- | --- | --- | --- |
| REV-0145284 | 823220 | 1863 | 57372 | 2022-10-06 | 5 | Highly recommend |
| REV-0104936 | 584588 | 242 | 75635 | 2018-06-04 | 5 | Great quality |
| REV-0053494 | 295066 | 976 | 64604 | 2015-07-19 | 4 | Happy with purchase |

---

## 📈 Analytical — Phân tích

### sales.csv / sales_test.csv — Dữ liệu doanh thu

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `Date` | date | Ngày đặt hàng |
| `Revenue` | float | Tổng doanh thu thuần |
| `COGS` | float | Tổng giá vốn hàng bán |


#### Ví dụ dữ liệu thật (sales.csv)

| Date | Revenue | COGS |
| --- | --- | --- |
| 2019-12-17 | 1509802.09 | 1466889.03 |
| 2013-07-07 | 2958215.01 | 2950992.63 |
| 2018-05-29 | 10262818.32 | 7905889.83 |

### sample_submission.csv — Định dạng file nộp bài mẫu


#### Ví dụ dữ liệu thật (sample_submission.csv)

| Date | Revenue | COGS |
| --- | --- | --- |
| 2024-06-18 | 4425117.3 | 3555475.26 |
| 2024-06-06 | 3060355.01 | 2455309.4 |
| 2024-03-18 | 3897806.29 | 3214985.47 |

---

## ⚙️ Operational — Vận hành

### inventory.csv — Tồn kho cuối tháng

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `snapshot_date` | date | Ngày chụp (cuối tháng) |
| `product_id` | int | FK → products |
| `stock_on_hand` | int | Tồn kho cuối tháng |
| `units_received` | int | Số lượng nhập kho |
| `units_sold` | int | Số lượng bán ra |
| `stockout_days` | int | Số ngày hết hàng |
| `days_of_supply` | float | Số ngày tồn kho đáp ứng được |
| `fill_rate` | float | Tỷ lệ đơn được đáp ứng đủ |
| `stockout_flag` | int | Cờ hết hàng |
| `overstock_flag` | int | Cờ tồn kho vượt mức |
| `reorder_flag` | int | Cờ cần tái đặt hàng |
| `sell_through_rate` | float | Tỷ lệ hàng đã bán / tổng sẵn có |
| `product_name` | str | Tên sản phẩm (denormalized) |
| `category` | str | Danh mục sản phẩm (denormalized) |
| `segment` | str | Phân khúc thị trường (denormalized) |
| `year` | int | Năm |
| `month` | int | Tháng |


#### Ví dụ dữ liệu thật (inventory.csv)

| snapshot_date | product_id | stock_on_hand | units_received | units_sold | stockout_days | days_of_supply | fill_rate | stockout_flag | overstock_flag | reorder_flag | sell_through_rate | product_name | category | segment | year | month |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015-05-31 | 1119 | 48 | 33 | 30 | 1 | 48.0 | 0.9667 | 1 | 0 | 0 | 0.3846 | MekongFit RS-07 | Outdoor | Premium | 2015 | 5 |
| 2015-02-28 | 1818 | 212 | 14 | 13 | 2 | 489.2 | 0.9333 | 1 | 1 | 0 | 0.0578 | SaigonCore YY-13 | GenZ | Trendy | 2015 | 2 |
| 2013-02-28 | 647 | 187 | 21 | 18 | 1 | 311.7 | 0.9667 | 1 | 1 | 0 | 0.0878 | SaigonFlex UC-12 | Streetwear | Everyday | 2013 | 2 |

### web_traffic.csv — Lưu lượng website hàng ngày

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `date` | date | Ngày ghi nhận |
| `sessions` | int | Tổng số phiên truy cập |
| `unique_visitors` | int | Khách truy cập duy nhất |
| `page_views` | int | Tổng lượt xem trang |
| `bounce_rate` | float | Tỷ lệ thoát |
| `avg_session_duration_sec` | float | Thời gian trung bình / phiên (giây) |
| `traffic_source` | str | Kênh nguồn traffic |


#### Ví dụ dữ liệu thật (web_traffic.csv)

| date | sessions | unique_visitors | page_views | bounce_rate | avg_session_duration_sec | traffic_source |
| --- | --- | --- | --- | --- | --- | --- |
| 2015-07-23 | 25169 | 19631 | 112460 | 0.00453 | 134.2 | direct |
| 2015-09-12 | 15341 | 11807 | 51281 | 0.00518 | 277.5 | direct |
| 2022-02-02 | 27306 | 21266 | 126792 | 0.0035 | 162.0 | organic_search |

---

## Quan hệ giữa các bảng

| Quan hệ | Cardinality |
|---------|-------------|
| orders ↔ payments | 1 : 1 |
| orders ↔ shipments | 1 : 0 hoặc 1 |
| orders ↔ returns | 1 : 0 hoặc nhiều |
| orders ↔ reviews | 1 : 0 hoặc nhiều (~20% đơn delivered) |
| order_items ↔ promotions | nhiều : 0 hoặc 1 |
| products ↔ inventory | 1 : nhiều (1 dòng/sản phẩm/tháng) |
