"""Create an annotated copy of the Phase 2 EDA notebook.

The source notebook keeps the full executable analysis. This script adds short
business-reading markdown notes after the main output cells so the notebook is
easier to present and review.

Run from repo root:
    uv run python group_job/build_annotated_phase2_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "group_job" / "phase2_product_insights_eda.ipynb"
DST = ROOT / "group_job" / "phase2_product_insights_eda_annotated.ipynb"


ANNOTATION_TAG = "phase2_short_business_note"


def note(text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(text.strip() + "\n")
    cell.metadata["tags"] = [ANNOTATION_TAG]
    return cell


notes_by_source_start = {
    'feasibility = pd.DataFrame(': """
        **Cách đọc.** Bảng này là bộ lọc câu hỏi EDA: câu nào dữ liệu hỗ trợ thì phân tích, câu nào thiếu biến như TikTok/platform chi tiết hoặc marketing spend thì không kết luận quá mức.
    """,
    'return_product = (': """
        **Cách đọc.** `revenue`, `gross_margin`, promotion share, return rate và review được gom về cấp sản phẩm. Đây là bảng nền để so sánh "bán chạy", "có lợi nhuận", "phụ thuộc promotion" và "rủi ro hoàn hàng".
    """,
    'top_margin = product_summary.sort_values("gross_margin", ascending=False).head(12)': """
        **Nhận xét.** Hai bảng này cho thấy best-seller không nhất thiết là profit driver. Nên ưu tiên sản phẩm theo gross margin và margin rate, rồi kiểm tra thêm promotion dependency và return rate trước khi scale.
    """,
    'plot_df = top_margin.sort_values("gross_margin")': """
        **Nhận xét chart.** Mỗi điểm là một sản phẩm: trục X là số lượng bán, trục Y là revenue, màu là category, kích thước là gross margin rate. Điểm chính: bán nhiều chưa chắc tạo doanh thu/lợi nhuận cao.
    """,
    'category_month = (': """
        **Nhận xét heatmap.** Seasonality index > 1 nghĩa là tháng đó bán cao hơn trung bình của chính category đó. Streetwear/Casual/GenZ mạnh vào khoảng tháng 4-6, còn Outdoor peak rõ hơn ở tháng 12.
    """,
    'product_month = (': """
        **Nhận xét.** Bảng này đưa seasonality xuống cấp sản phẩm. `peak_month` giúp biết tháng vàng của từng product, còn `peak_season_index` cho biết peak đó mạnh hơn tháng trung bình bao nhiêu lần.
    """,
    'source_summary = (': """
        **Nhận xét kênh đặt hàng.** `order_source` cho biết khách đến từ đâu để đặt hàng. Channel mix giữa category khá giống nhau: organic search thường là nguồn lớn nhất, sau đó là paid search và social media. Không nên suy ra TikTok riêng vì dữ liệu chỉ có `social_media` tổng quát.
    """,
    'min_revenue = product_summary["revenue"].quantile(0.75)': """
        **Nhận xét promotion.** Bảng trên là sản phẩm revenue/volume lớn nhưng phụ thuộc promotion cao; cần guardrail margin vì có sản phẩm doanh thu lớn nhưng margin âm. Bảng dưới là nhóm bán tốt nhưng ít phụ thuộc promotion hơn, đáng ưu tiên scale bằng visibility/inventory thay vì giảm giá sâu.
    """,
    'promo_fact = fact[fact["promo_id"].notna()].merge(': """
        **Nhận xét promo channel.** `promo_channel` khác `order_source`: đây là kênh phân phối khuyến mãi. `avg_discount` là số tiền giảm trung bình mỗi dòng item, không phải %. Online promotion tạo revenue lớn nhưng margin âm, nên cần audit discount/product mix.
    """,
    'returns_with_category = returns.merge(products[["product_id", "product_name", "category", "segment"]]': """
        **Nhận xét return/review.** `wrong_size` là lý do hoàn hàng lớn nhất. Rating trung bình gần như không giải thích được return rate, nên không thể chỉ nhìn review; cần xử lý theo reason cụ thể như size, defect và mô tả sản phẩm.
    """,
    'fact_geo = fact.merge(geography[["zip", "city", "region", "district"]]': """
        **Nhận xét location.** Region/city cho biết demand tập trung ở đâu: Streetwear và GenZ nghiêng về East, Casual và Outdoor nghiêng về West. Tuy nhiên inventory không có region/warehouse, nên chỉ dùng cho campaign targeting, không kết luận tái phân bổ kho vùng.
    """,
    'inv_month = (': """
        **Nhận xét inventory readiness.** Cell này nối tháng peak của sản phẩm với tồn kho tháng peak và tháng trước peak. `inventory_risk_score` càng cao nghĩa là vừa có stockout nhiều vừa fill rate thấp quanh mùa bán mạnh.
    """,
    'inv_product = (': """
        **Nhận xét overstock.** Chart overstock map tìm sản phẩm có days of supply cao nhưng sell-through thấp. Vùng đáng lo là bên phải + phía dưới + điểm to: tồn nhiều, đủ bán rất lâu, nhưng quay vòng chậm.
    """,
    'top_cat = category_summary.sort_values("gross_margin", ascending=False).iloc[0]': """
        **Cách dùng cho report.** Mạch kể chuyện nên đi từ profit pool -> seasonality -> promotion/channel -> return leakage -> location demand -> inventory action. Các claim đều đã có bảng/figure hỗ trợ trong notebook.
    """,
}


nb = nbf.read(SRC, as_version=4)

# Remove older generated notes if this script is rerun.
clean_cells = []
for cell in nb.cells:
    if cell.cell_type == "markdown" and ANNOTATION_TAG in cell.metadata.get("tags", []):
        continue
    clean_cells.append(cell)

new_cells = []
for cell in clean_cells:
    new_cells.append(cell)
    if cell.cell_type != "code":
        continue
    source = cell.source.lstrip()
    for prefix, text in notes_by_source_start.items():
        if source.startswith(prefix):
            new_cells.append(note(text))
            break

nb.cells = new_cells
nbf.write(nb, DST)
print(f"Wrote {DST}")
