### Executive synthesis

1. **Profit pool tập trung ở sản phẩm/danh mục khác với volume pool.** `Streetwear` là category dẫn đầu gross margin (1.74B), còn sản phẩm margin cao nhất là `SaigonFlex UM-43` (130.46M). Top sản phẩm theo units không nên mặc định là ưu tiên scale nếu margin thấp hoặc âm.

2. **Seasonality đủ mạnh để lên lịch inventory/promotion theo category.** Peak month theo category: GenZ: tháng 6 (1.75x), Streetwear: tháng 5 (1.55x), Casual: tháng 5 (1.51x), Outdoor: tháng 12 (1.38x). Đây là cơ sở để đặt hàng trước mùa cao điểm và kiểm soát discount theo mùa.

3. **Promotion là đòn bẩy lớn nhưng cần guardrail margin.** Các dòng item có promotion đóng góp 33.1% revenue. `HanoiStreet RP-82` thuộc nhóm volume/revenue lớn có promo revenue share 51.0%; nếu margin thấp, promotion nên được dùng có mục tiêu thay vì scale đại trà.

4. **Return leakage nên xử lý theo nguyên nhân, không chỉ rating.** Lý do return lớn nhất là `wrong_size` (35.0%). Product risk cao nhất trong nhóm volume lớn là `UrbanVN RP-12` với return rate 6.1% và refund share 5.7%.

5. **Demand theo vùng có thể guide campaign, nhưng chưa đủ để quyết định kho vùng.** `orders.zip` + `geography` cho thấy demand concentration theo region/city; tuy nhiên inventory không có dimension region/warehouse.

6. **Inventory có hai bài toán song song: bảo vệ peak demand và giảm overstock.** Sản phẩm overstock nổi bật là `HanoiStreet RP-79` với days of supply trung bình 2024.5 và sell-through 10.1%. Nhóm high-demand stockout nên được ưu tiên replenishment trước peak month.

### Cách đưa vào report

- Mở bằng nghịch lý: "best seller chưa chắc là profit driver".
- Sau đó nối sang seasonality: profit driver cần được bảo vệ bằng inventory đúng tháng.
- Tiếp theo là channel/promotion: scale channel nào và sản phẩm nào cần guardrail discount.
- Kết thúc bằng operational actions: giảm return theo nguyên nhân, xử lý overstock, và tạo feature cho forecasting Phase 3.
