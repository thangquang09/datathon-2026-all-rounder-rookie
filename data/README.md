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

### geography.csv — Địa lý

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `zip` | int | Khoá chính |
| `city` | str | Thành phố |
| `region` | str | Vùng địa lý |
| `district` | str | Quận/huyện |

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

### payments.csv — Thanh toán (quan hệ 1:1 với orders)

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | FK → orders |
| `payment_method` | str | Phương thức thanh toán |
| `payment_value` | float | Tổng giá trị thanh toán |
| `installments` | int | Số kỳ trả góp |

### shipments.csv — Vận chuyển (chỉ cho đơn shipped/delivered/returned)

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | FK → orders |
| `ship_date` | date | Ngày gửi hàng |
| `delivery_date` | date | Ngày giao hàng |
| `shipping_fee` | float | Phí vận chuyển |

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

---

## 📈 Analytical — Phân tích

### sales.csv / sales_test.csv — Dữ liệu doanh thu

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `Date` | date | Ngày đặt hàng |
| `Revenue` | float | Tổng doanh thu thuần |
| `COGS` | float | Tổng giá vốn hàng bán |

### sample_submission.csv — Định dạng file nộp bài mẫu

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
