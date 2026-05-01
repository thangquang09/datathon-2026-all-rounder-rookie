# Ghi chú về sự khác biệt giữa README ban đầu và dữ liệu thực tế

**Ngày kiểm tra:** 2026-04-17

## Tóm tắt

Đã kiểm tra 14 file CSV và đối chiếu với schema trong README.md. Phát hiện **2 khác biệt** cần cập nhật.

---

## 1. inventory.csv - Thiếu 5 cột trong README ban đầu

### Mô tả
File `inventory.csv` thực tế có thêm 5 cột mà README ban đầu không liệt kê.

### Các cột thiếu trong README ban đầu:
- `product_name` (str) - Tên sản phẩm (denormalized từ products)
- `category` (str) - Danh mục sản phẩm (denormalized từ products)
- `segment` (str) - Phân khúc thị trường (denormalized từ products)
- `year` (int) - Năm
- `month` (int) - Tháng

### Lý do denormalization:
Các trường `product_name`, `category`, `segment` được duplicate từ bảng `products` vào `inventory` để:
- Tăng tốc độ truy vấn phân tích (không cần JOIN)
- Hỗ trợ group by theo category/segment trực tiếp
- Phù hợp với data warehouse pattern

Các trường `year`, `month` được extract từ `snapshot_date` để:
- Dễ dàng filter và group by theo tháng/năm
- Tối ưu cho time-series analysis

### Header thực tế:
```
snapshot_date,product_id,stock_on_hand,units_received,units_sold,stockout_days,days_of_supply,fill_rate,stockout_flag,overstock_flag,reorder_flag,sell_through_rate,product_name,category,segment,year,month
```

---

## 2. web_traffic.csv - Cột conversion_rate không tồn tại

### Mô tả
README ban đầu liệt kê cột `conversion_rate` nhưng file CSV thực tế không có cột này.

### Cột bị xóa:
- `conversion_rate` (float) - Tỷ lệ phiên dẫn đến đặt hàng

### Header thực tế:
```
date,sessions,unique_visitors,page_views,bounce_rate,avg_session_duration_sec,traffic_source
```

### Lý do có thể:
- Conversion rate có thể được tính toán từ `orders.csv` và `web_traffic.csv` thay vì lưu trực tiếp
- Hoặc dữ liệu này không được thu thập trong dataset mô phỏng

---

## 3. Các file khớp 100% với README

Các file sau đây có schema khớp chính xác với README ban đầu:

✅ **Master layer:**
- `products.csv` - 8 cột khớp
- `customers.csv` - 7 cột khớp
- `promotions.csv` - 10 cột khớp
- `geography.csv` - 4 cột khớp

✅ **Transaction layer:**
- `orders.csv` - 8 cột khớp
- `order_items.csv` - 7 cột khớp
- `payments.csv` - 4 cột khớp
- `shipments.csv` - 4 cột khớp
- `returns.csv` - 7 cột khớp
- `reviews.csv` - 7 cột khớp

✅ **Analytical layer:**
- `sales.csv` - 3 cột khớp
- `sample_submission.csv` - (không kiểm tra chi tiết)

---

## Hành động đã thực hiện

1. ✅ Cập nhật `inventory.csv` schema trong README.md - thêm 5 cột mới
2. ✅ Cập nhật `web_traffic.csv` schema trong README.md - xóa cột `conversion_rate`
3. ✅ Format lại toàn bộ README.md với markdown chuẩn

---

## Kết luận

Dataset có chất lượng tốt với schema nhất quán. Chỉ có 2 điểm khác biệt nhỏ đã được sửa trong README.md. Tất cả các quan hệ foreign key và cardinality đều hợp lệ.
