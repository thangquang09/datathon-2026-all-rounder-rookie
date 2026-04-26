"""Create an annotated copy of the Phase 2 EDA notebook.

The original notebook is left untouched. This script inserts concise markdown
interpretation cells after the main analytical outputs.

Run from the repository root:
    uv run python group_job/create_annotated_phase2_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "group_job" / "phase2_product_insights_eda.ipynb"
DST = ROOT / "group_job" / "phase2_product_insights_eda_annotated.ipynb"


ANNOTATIONS = {
    "DATE_COLS = {": """
### Nhận xét

Notebook load toàn bộ bảng chính và kiểm tra coverage/missingness trước khi phân tích. Đây là bước guardrail: nếu bảng thiếu ngày, thiếu join key, hoặc date range lệch nhau thì các insight phía sau dễ sai. Các phân tích bên dưới chỉ dùng dữ liệu nội bộ trong `data/`, đúng constraint cuộc thi.
""",
    "feasibility = pd.DataFrame": """
### Nhận xét

Bảng này chuyển draft thành câu hỏi EDA khả thi. Điểm quan trọng là không cố kết luận phần dữ liệu không hỗ trợ: không có TikTok/platform chi tiết, không có marketing spend, và không có experiment/control group để kết luận causal lift của promotion. Vì vậy notebook chỉ kết luận ở mức mô tả/diagnostic/prescriptive có căn cứ.
""",
    "daily_from_fact = (": """
### Nhận xét

`line_revenue = quantity * unit_price` khớp với `sales.csv` theo ngày, nên có thể dùng bảng giao dịch dòng item để phân tích sản phẩm mà không bị lệch tổng doanh thu. Đây là kiểm tra quan trọng nhất trước khi đi vào product-level EDA.
""",
    "category_summary = (": """
### Nhận xét

Bảng category/product summary tạo các metric nền: units, revenue, gross margin, promo share, return rate và review. Các metric này giúp tránh nhìn một chiều: sản phẩm bán nhiều chưa chắc có margin tốt, sản phẩm revenue cao có thể phụ thuộc promotion hoặc bị return nhiều.
""",
    "top_margin = product_summary": """
### Nhận xét

Hai bảng top product cho thấy **best seller không đồng nghĩa profit driver**. Top theo units và top theo gross margin chỉ trùng một phần; có sản phẩm bán rất nhiều nhưng margin thấp hoặc âm. Khi chọn sản phẩm để scale, nên ưu tiên tổ hợp `gross_margin`, `gross_margin_rate`, `promo_revenue_share` và `return_rate_qty`, không chỉ `units`.
""",
    "plot_df = top_margin.sort_values": """
### Nhận xét

Biểu đồ scatter đọc như sau: trục X là số lượng bán, trục Y là revenue, màu là category, kích thước điểm là gross margin rate. Nếu một điểm rất xa bên phải nhưng không cao hoặc điểm nhỏ, sản phẩm đó bán nhiều nhưng chưa chắc tạo revenue/margin tốt. Các nhãn top margin giúp thấy profit drivers không nhất thiết là sản phẩm bán nhiều nhất.
""",
    "category_month = (": """
### Nhận xét

Seasonality index = revenue tháng đó / revenue trung bình tháng của chính category. Giá trị `> 1` là tháng mạnh hơn bình quân. Heatmap cho thấy Streetwear, Casual và GenZ mạnh ở giai đoạn tháng 4-6, trong khi Outdoor peak rõ hơn vào tháng 12. Vì vậy lịch inventory/promotion nên tách theo category.
""",
    "product_month = (": """
### Nhận xét

Bảng peak month ở cấp sản phẩm giúp cụ thể hóa seasonality: mỗi product có tháng vàng và mức peak riêng. Insight này dùng được cho replenishment và campaign planning: sản phẩm có `peak_season_index` cao cần được chuẩn bị tồn kho trước peak month, không chỉ nhìn trung bình năm.
""",
    "source_summary = (": """
### Nhận xét

`order_source` cho biết nguồn đặt hàng. Heatmap category-source cho thấy mix kênh giữa các category khá giống nhau: organic search lớn nhất, sau đó paid search và social media. Vì vậy chưa có bằng chứng mạnh để nói một category phụ thuộc riêng một kênh; muốn tối ưu ROI cần thêm cost/spend theo channel.
""",
    "promo_dependent = (": """
### Nhận xét

Hai bảng này tách sản phẩm thành nhóm phụ thuộc promotion cao và nhóm bán tốt nhưng ít phụ thuộc promotion. Nhóm promo-dependent cần kiểm tra margin guardrail: có sản phẩm tạo revenue lớn nhưng gross margin âm. Nhóm ít phụ thuộc promotion nhưng margin cao là candidate tốt hơn để scale bền vững bằng visibility/inventory thay vì discount sâu.
""",
    "promo_fact = fact[fact[\"promo_id\"].notna()]": """
### Nhận xét

`promo_channel` khác `order_source`: đây là kênh phân phối chương trình khuyến mãi, không phải nguồn khách vào đặt hàng. Bảng cho thấy kênh tạo revenue lớn chưa chắc tạo margin tốt; đặc biệt nếu một promo channel có gross margin rate âm, cần audit discount/product mix trước khi scale campaign.
""",
    "returns_with_category = returns.merge": """
### Nhận xét

Return analysis cho thấy lý do hoàn hàng lớn nhất là `wrong_size`, sau đó là `defective` và `not_as_described`. Rating trung bình gần như không tương quan với return rate, nên không thể dùng review score thay thế phân tích return. Hành động nên tập trung vào size guide, QC và mô tả/ảnh sản phẩm.
""",
    "fact_geo = fact.merge": """
### Nhận xét

Location analysis dùng `zip -> city/region` để tìm demand concentration. Streetwear và GenZ nghiêng về East, còn Casual và Outdoor nghiêng về West. Tuy nhiên inventory không có region/warehouse, nên chỉ nên dùng insight này cho campaign targeting hoặc regional merchandising, không kết luận tái phân bổ kho theo vùng.
""",
    "inv_month = (": """
### Nhận xét

Cell này nối seasonality với inventory. `inventory_risk_score` tăng khi sản phẩm bị stockout trong tháng peak/tháng trước peak hoặc fill rate thấp. Bảng giúp tìm product có margin/demand cao nhưng readiness kém, tức có nguy cơ mất doanh thu đúng mùa mạnh nhất.
""",
    "inv_product = (": """
### Nhận xét

Overstock map tìm sản phẩm có `days_of_supply` cao nhưng `sell_through_rate` thấp. Vùng rủi ro là bên phải và phía dưới biểu đồ, đặc biệt các điểm lớn vì đang giữ nhiều tồn kho. Nhóm này nên giảm reorder, dùng clearance/bundle có kiểm soát margin, hoặc rà lại forecast/replenishment.
""",
    "top_cat = category_summary": """
### Nhận xét

Phần synthesis gom các insight thành câu chuyện Phase 2: profit pool khác volume pool, seasonality khác theo category/product, promotion cần guardrail margin, return leakage cần xử lý theo reason, location chỉ đủ cho demand targeting, và inventory cần vừa bảo vệ peak demand vừa giảm overstock.
""",
}


def annotation_for(cell_source: str) -> str | None:
    for key, note in ANNOTATIONS.items():
        if key in cell_source:
            return note.strip() + "\n"
    return None


def main() -> None:
    nb = nbf.read(SRC, as_version=4)
    new_cells = []
    for cell in nb.cells:
        new_cells.append(cell)
        if cell.cell_type == "code":
            note = annotation_for(cell.source)
            if note:
                new_cells.append(nbf.v4.new_markdown_cell(note))

    nb.cells = new_cells
    nbf.write(nb, DST)
    print(f"Wrote {DST}")
    print(f"Original cells: {len(nbf.read(SRC, as_version=4).cells)}")
    print(f"Annotated cells: {len(nb.cells)}")


if __name__ == "__main__":
    main()
