"""Build the Phase 2 product insight EDA notebook.

Run from the repository root:
    uv run python group_job/build_phase2_product_insights_notebook.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "phase2_product_insights_eda.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(src).strip() + "\n")


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(src).strip() + "\n")


cells: list[nbf.NotebookNode] = []


cells.append(
    md(
        """
        # Phase 2 EDA - Product, Promotion, Return, Inventory Insights

        Workspace: `group_job`

        Mục tiêu notebook này là biến draft câu hỏi thành các phân tích có thể kiểm chứng bằng dữ liệu. Trọng tâm là sản phẩm: sản phẩm nào đóng góp doanh thu/lợi nhuận, bán tốt vào thời điểm nào, phụ thuộc promotion ra sao, được review/trả hàng như thế nào, khu vực nào mua nhiều, và tồn kho có sẵn trước các tháng cao điểm hay không.

        Notebook chỉ dùng dữ liệu được cung cấp trong `data/`. Các góc không có dữ liệu hỗ trợ, ví dụ TikTok cụ thể, marketing spend, hoặc causal ROI của promotion, được ghi rõ là không kết luận.
        """
    )
)


cells.append(
    md(
        """
        ## 0. Từ draft sang câu hỏi EDA khả thi

        Draft ban đầu xoay quanh chuỗi: **top sản phẩm đóng góp lợi nhuận -> bán khi nào -> bán qua kênh nào -> bán qua marketing/promotion nào -> tối ưu season, platform, promotion cost**.

        Trong dữ liệu có thể làm tốt các câu hỏi sau:

        1. Sản phẩm/danh mục nào đóng góp nhiều nhất vào doanh thu và gross margin? Top bán chạy theo số lượng có khác top lợi nhuận không?
        2. Mỗi danh mục và nhóm sản phẩm chủ lực đạt "thời gian vàng" vào tháng nào?
        3. Doanh thu/lợi nhuận đến từ kênh đặt hàng nào (`order_source`) và mức phụ thuộc promotion của từng sản phẩm là bao nhiêu?
        4. Sản phẩm nào chỉ bán mạnh khi có promotion, nhưng margin thấp hoặc âm?
        5. Sản phẩm/danh mục nào bị rò rỉ doanh thu qua hoàn hàng/refund, và review có cảnh báo gì không?
        6. Khu vực nào thường mua từng danh mục/sản phẩm?
        7. Tồn kho có sẵn trong/tháng trước mùa cao điểm không? Sản phẩm nào vừa stockout vừa có nhu cầu cao, và sản phẩm nào tồn lâu/tồn nhiều?

        Các câu hỏi **không kết luận cứng** vì thiếu dữ liệu:

        - "Sản phẩm lợi nhuận cao đang được bán trên TikTok không?" Không có trường TikTok/platform cụ thể; chỉ có `order_source`, `promo_channel`, `traffic_source`.
        - "Tối ưu chi phí marketing/promotion ROI" Không có marketing spend hoặc campaign cost.
        - "Promotion gây ra tăng trưởng bao nhiêu?" Dữ liệu quan sát không có nhóm control/randomization; notebook chỉ đo phụ thuộc promotion và pattern trước/sau mô tả.
        """
    )
)


cells.append(md("## 1. Setup, load dữ liệu, kiểm tra coverage"))

cells.append(
    code(
        """
        from __future__ import annotations

        import json
        import warnings
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display

        warnings.filterwarnings("ignore")

        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 160
        plt.rcParams["savefig.bbox"] = "tight"
        pd.options.display.max_columns = 80
        pd.options.display.float_format = "{:,.4f}".format

        ROOT = Path.cwd().resolve()
        while not (ROOT / "data").is_dir() and ROOT != ROOT.parent:
            ROOT = ROOT.parent

        DATA = ROOT / "data"
        WORK = ROOT / "group_job"
        FIG = WORK / "figures"
        TABLE = WORK / "tables"
        FIG.mkdir(parents=True, exist_ok=True)
        TABLE.mkdir(parents=True, exist_ok=True)

        def savefig(name: str) -> Path:
            path = FIG / name
            plt.tight_layout()
            plt.savefig(path)
            plt.show()
            return path

        def fmt_money(v: float) -> str:
            if pd.isna(v):
                return ""
            av = abs(float(v))
            if av >= 1e9:
                return f"{v / 1e9:,.2f}B"
            if av >= 1e6:
                return f"{v / 1e6:,.2f}M"
            if av >= 1e3:
                return f"{v / 1e3:,.2f}K"
            return f"{v:,.2f}"

        def fmt_pct(v: float) -> str:
            if pd.isna(v):
                return ""
            return f"{100 * float(v):.1f}%"

        def view_table(df: pd.DataFrame, money_cols=(), pct_cols=(), int_cols=(), n: int | None = None) -> pd.DataFrame:
            out = df.head(n).copy() if n else df.copy()
            for col in money_cols:
                if col in out:
                    out[col] = out[col].map(fmt_money)
            for col in pct_cols:
                if col in out:
                    out[col] = out[col].map(fmt_pct)
            for col in int_cols:
                if col in out:
                    out[col] = out[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
            return out

        print("Repo root:", ROOT)
        print("Data dir :", DATA)
        print("Output   :", WORK)
        """
    )
)


cells.append(
    code(
        """
        DATE_COLS = {
            "orders": ["order_date"],
            "customers": ["signup_date"],
            "promotions": ["start_date", "end_date"],
            "shipments": ["ship_date", "delivery_date"],
            "returns": ["return_date"],
            "reviews": ["review_date"],
            "inventory": ["snapshot_date"],
            "web_traffic": ["date"],
            "sales": ["Date"],
            "sample_submission": ["Date"],
        }

        DTYPE = {
            "order_items": {"promo_id": "string", "promo_id_2": "string"},
            "promotions": {"promo_id": "string"},
        }

        def load_csv(name: str) -> pd.DataFrame:
            return pd.read_csv(
                DATA / f"{name}.csv",
                parse_dates=DATE_COLS.get(name, []),
                dtype=DTYPE.get(name),
                low_memory=False,
            )

        sales = load_csv("sales")
        orders = load_csv("orders")
        order_items = load_csv("order_items")
        products = load_csv("products")
        customers = load_csv("customers")
        promotions = load_csv("promotions")
        payments = load_csv("payments")
        shipments = load_csv("shipments")
        returns = load_csv("returns")
        reviews = load_csv("reviews")
        geography = load_csv("geography")
        inventory = load_csv("inventory")
        web = load_csv("web_traffic")

        tables = {
            "sales": sales,
            "orders": orders,
            "order_items": order_items,
            "products": products,
            "customers": customers,
            "promotions": promotions,
            "payments": payments,
            "shipments": shipments,
            "returns": returns,
            "reviews": reviews,
            "geography": geography,
            "inventory": inventory,
            "web_traffic": web,
        }

        summary_rows = []
        for name, df in tables.items():
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            summary_rows.append(
                {
                    "table": name,
                    "rows": len(df),
                    "cols": df.shape[1],
                    "date_min": min((df[c].min() for c in date_cols), default=pd.NaT),
                    "date_max": max((df[c].max() for c in date_cols), default=pd.NaT),
                    "max_missing_pct": df.isna().mean().max() if len(df.columns) else 0,
                }
            )

        data_summary = pd.DataFrame(summary_rows).sort_values("rows", ascending=False)
        display(view_table(data_summary, pct_cols=["max_missing_pct"], int_cols=["rows", "cols"]))
        data_summary.to_csv(TABLE / "00_data_summary.csv", index=False)
        """
    )
)


cells.append(
    code(
        """
        feasibility = pd.DataFrame(
            [
                {
                    "question": "Top sản phẩm theo doanh thu, số lượng, gross margin",
                    "sources": "order_items + orders + products + sales",
                    "verdict": "Làm được",
                    "caveat": "Gross margin dùng unit_price sau giảm giá trừ product cogs; chưa trừ marketing/logistics.",
                },
                {
                    "question": "Bán chạy/lợi nhuận cao vào thời điểm nào",
                    "sources": "orders.order_date + order_items + products",
                    "verdict": "Làm được",
                    "caveat": "Phân tích theo tháng/ngày; inventory chỉ có snapshot cuối tháng.",
                },
                {
                    "question": "Bán qua kênh nào",
                    "sources": "orders.order_source, web_traffic.traffic_source",
                    "verdict": "Làm được",
                    "caveat": "order_source không phải platform social chi tiết.",
                },
                {
                    "question": "Sản phẩm nào phụ thuộc promotion",
                    "sources": "order_items.promo_id/promo_id_2 + promotions",
                    "verdict": "Làm được mô tả",
                    "caveat": "Không suy ra causal lift nếu không có control/randomization.",
                },
                {
                    "question": "Tối ưu chi phí marketing/promotion",
                    "sources": "Không có spend/cost campaign",
                    "verdict": "Không kết luận ROI",
                    "caveat": "Chỉ phân tích revenue/margin/discount/refund theo kênh.",
                },
                {
                    "question": "Review và hoàn hàng theo sản phẩm",
                    "sources": "reviews + returns + products",
                    "verdict": "Làm được",
                    "caveat": "Review text chỉ có title ngắn; dùng rating và title category nhẹ.",
                },
                {
                    "question": "Khu vực thường mua sản phẩm/danh mục",
                    "sources": "orders.zip + geography + order_items + products",
                    "verdict": "Làm được",
                    "caveat": "Không có tồn kho theo kho/khu vực, chỉ có demand region.",
                },
                {
                    "question": "Có đủ hàng trước thời gian vàng, hàng tồn lâu/tồn nhiều",
                    "sources": "inventory monthly + product demand by month",
                    "verdict": "Làm được ở mức tháng",
                    "caveat": "Không thấy tồn kho theo ngày/kho; đánh giá bằng stockout_days/fill_rate/days_of_supply.",
                },
                {
                    "question": "Sản phẩm đang bán trên TikTok để scale platform",
                    "sources": "Không có TikTok/platform granular",
                    "verdict": "Không làm",
                    "caveat": "Có thể thay bằng social_media/order_source hoặc promo_channel=social_media.",
                },
            ]
        )
        display(feasibility)
        feasibility.to_csv(TABLE / "01_question_feasibility.csv", index=False)
        """
    )
)


cells.append(md("## 2. Tạo fact table sản phẩm và kiểm tra với `sales.csv`"))

cells.append(
    code(
        """
        fact = (
            order_items.reset_index(names="line_id")
            .merge(
                orders[
                    [
                        "order_id",
                        "order_date",
                        "customer_id",
                        "zip",
                        "order_status",
                        "payment_method",
                        "device_type",
                        "order_source",
                    ]
                ],
                on="order_id",
                how="left",
                validate="many_to_one",
            )
            .merge(products, on="product_id", how="left", validate="many_to_one")
        )

        fact["line_revenue"] = fact["quantity"] * fact["unit_price"]
        fact["line_gross_before_discount"] = fact["line_revenue"] + fact["discount_amount"].fillna(0)
        fact["line_cogs"] = fact["quantity"] * fact["cogs"]
        fact["gross_margin"] = fact["line_revenue"] - fact["line_cogs"]
        fact["promo_used"] = fact[["promo_id", "promo_id_2"]].notna().any(axis=1)
        fact["promo_revenue"] = np.where(fact["promo_used"], fact["line_revenue"], 0.0)
        fact["promo_units"] = np.where(fact["promo_used"], fact["quantity"], 0)
        fact["year"] = fact["order_date"].dt.year
        fact["month"] = fact["order_date"].dt.month
        fact["dow"] = fact["order_date"].dt.day_name()
        fact["ym"] = fact["order_date"].dt.to_period("M").dt.to_timestamp()

        daily_from_fact = (
            fact.groupby("order_date", as_index=False)
            .agg(item_revenue=("line_revenue", "sum"), item_cogs=("line_cogs", "sum"))
        )
        daily_check = sales.merge(daily_from_fact, left_on="Date", right_on="order_date", how="left")
        daily_check["revenue_abs_diff"] = (daily_check["Revenue"] - daily_check["item_revenue"]).abs()
        daily_check["cogs_abs_diff"] = (daily_check["COGS"] - daily_check["item_cogs"]).abs()

        print("Fact rows:", f"{len(fact):,}")
        print("Missing order join:", int(fact["order_date"].isna().sum()))
        print("Missing product join:", int(fact["product_name"].isna().sum()))
        print("Max revenue diff vs sales.csv:", daily_check["revenue_abs_diff"].max())
        print("Max COGS diff vs sales.csv:", daily_check["cogs_abs_diff"].max())
        print("Revenue correlation vs sales.csv:", daily_check[["Revenue", "item_revenue"]].corr().iloc[0, 1])

        display(
            Markdown(
                f'''
                **Data check.** Tổng `line_revenue = quantity * unit_price` khớp `sales.csv` theo ngày với sai số lớn nhất `{daily_check['revenue_abs_diff'].max():,.6f}`.
                Vì vậy các phân tích sản phẩm bên dưới dùng `line_revenue` làm revenue source of truth ở cấp dòng hàng.
                '''
            )
        )
        """
    )
)


cells.append(
    code(
        """
        return_product = (
            returns.groupby("product_id", as_index=False)
            .agg(
                return_qty=("return_quantity", "sum"),
                refund_amount=("refund_amount", "sum"),
                return_orders=("order_id", "nunique"),
                first_return=("return_date", "min"),
                last_return=("return_date", "max"),
            )
        )

        review_product = (
            reviews.groupby("product_id", as_index=False)
            .agg(avg_rating=("rating", "mean"), n_reviews=("rating", "size"))
        )

        product_summary = (
            fact.groupby(["product_id", "product_name", "category", "segment", "size", "color"], as_index=False)
            .agg(
                units=("quantity", "sum"),
                orders=("order_id", "nunique"),
                revenue=("line_revenue", "sum"),
                gross_margin=("gross_margin", "sum"),
                discount_amount=("discount_amount", "sum"),
                gross_before_discount=("line_gross_before_discount", "sum"),
                promo_revenue=("promo_revenue", "sum"),
                promo_units=("promo_units", "sum"),
            )
            .merge(return_product, on="product_id", how="left")
            .merge(review_product, on="product_id", how="left")
        )

        fill_zero = ["return_qty", "refund_amount", "return_orders", "promo_revenue", "promo_units"]
        product_summary[fill_zero] = product_summary[fill_zero].fillna(0)
        product_summary["gross_margin_rate"] = product_summary["gross_margin"] / product_summary["revenue"]
        product_summary["discount_rate"] = product_summary["discount_amount"] / product_summary["gross_before_discount"]
        product_summary["promo_revenue_share"] = product_summary["promo_revenue"] / product_summary["revenue"]
        product_summary["promo_unit_share"] = product_summary["promo_units"] / product_summary["units"]
        product_summary["return_rate_qty"] = product_summary["return_qty"] / product_summary["units"]
        product_summary["refund_share"] = product_summary["refund_amount"] / product_summary["revenue"]

        category_base = (
            fact.groupby("category", as_index=False)
            .agg(
                products=("product_id", "nunique"),
                units=("quantity", "sum"),
                orders=("order_id", "nunique"),
                revenue=("line_revenue", "sum"),
                gross_margin=("gross_margin", "sum"),
                discount_amount=("discount_amount", "sum"),
                promo_revenue=("promo_revenue", "sum"),
            )
        )
        category_returns = (
            returns.merge(products[["product_id", "category"]], on="product_id", how="left")
            .groupby("category", as_index=False)
            .agg(return_qty=("return_quantity", "sum"), refund_amount=("refund_amount", "sum"))
        )
        category_reviews = (
            reviews.merge(products[["product_id", "category"]], on="product_id", how="left")
            .groupby("category", as_index=False)
            .agg(n_reviews=("rating", "size"), avg_rating=("rating", "mean"))
        )
        category_summary = (
            category_base.merge(category_returns, on="category", how="left")
            .merge(category_reviews, on="category", how="left")
        )
        category_summary[["return_qty", "refund_amount", "n_reviews"]] = category_summary[
            ["return_qty", "refund_amount", "n_reviews"]
        ].fillna(0)
        category_summary["revenue_share"] = category_summary["revenue"] / category_summary["revenue"].sum()
        category_summary["gross_margin_rate"] = category_summary["gross_margin"] / category_summary["revenue"]
        category_summary["promo_revenue_share"] = category_summary["promo_revenue"] / category_summary["revenue"]
        category_summary["return_rate_qty"] = category_summary["return_qty"] / category_summary["units"]
        category_summary["refund_share"] = category_summary["refund_amount"] / category_summary["revenue"]

        display(
            view_table(
                category_summary.sort_values("gross_margin", ascending=False),
                money_cols=["revenue", "gross_margin", "discount_amount", "promo_revenue", "refund_amount"],
                pct_cols=["revenue_share", "gross_margin_rate", "promo_revenue_share", "return_rate_qty", "refund_share"],
                int_cols=["products", "units", "orders", "return_qty", "n_reviews"],
            )
        )
        category_summary.to_csv(TABLE / "02_category_summary.csv", index=False)
        product_summary.to_csv(TABLE / "03_product_summary.csv", index=False)
        """
    )
)


cells.append(md("## 3. Sản phẩm nào thật sự đóng góp lợi nhuận?"))

cells.append(
    code(
        """
        top_margin = product_summary.sort_values("gross_margin", ascending=False).head(12)
        top_units = product_summary.sort_values("units", ascending=False).head(12)
        top_revenue = product_summary.sort_values("revenue", ascending=False).head(12)

        display(Markdown("### Top 12 theo gross margin"))
        display(
            view_table(
                top_margin[
                    [
                        "product_name",
                        "category",
                        "segment",
                        "units",
                        "revenue",
                        "gross_margin",
                        "gross_margin_rate",
                        "promo_revenue_share",
                        "return_rate_qty",
                        "avg_rating",
                        "n_reviews",
                    ]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["gross_margin_rate", "promo_revenue_share", "return_rate_qty"],
                int_cols=["units", "n_reviews"],
            )
        )

        display(Markdown("### Top 12 theo số lượng bán"))
        display(
            view_table(
                top_units[
                    [
                        "product_name",
                        "category",
                        "segment",
                        "units",
                        "revenue",
                        "gross_margin",
                        "gross_margin_rate",
                        "promo_revenue_share",
                        "return_rate_qty",
                    ]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["gross_margin_rate", "promo_revenue_share", "return_rate_qty"],
                int_cols=["units"],
            )
        )

        overlap = len(set(top_margin["product_id"]) & set(top_units["product_id"]))
        margin_share_top10 = product_summary.nlargest(10, "gross_margin")["gross_margin"].sum() / product_summary["gross_margin"].sum()
        revenue_share_top10 = product_summary.nlargest(10, "revenue")["revenue"].sum() / product_summary["revenue"].sum()

        display(
            Markdown(
                f'''
                **Insight.** Top 12 theo số lượng và top 12 theo gross margin chỉ trùng `{overlap}` sản phẩm.
                Top 10 sản phẩm theo margin tạo `{fmt_pct(margin_share_top10)}` tổng gross margin, còn top 10 revenue tạo `{fmt_pct(revenue_share_top10)}` tổng revenue.
                Vì vậy "bán chạy" không đồng nghĩa "đáng scale"; cần xếp sản phẩm theo margin và rủi ro return/promotion.
                '''
            )
        )
        """
    )
)


cells.append(
    code(
        """
        plot_df = top_margin.sort_values("gross_margin")
        plt.figure(figsize=(11, 6))
        ax = sns.barplot(data=plot_df, y="product_name", x="gross_margin", hue="category", dodge=False)
        ax.set_title("Top sản phẩm theo gross margin")
        ax.set_xlabel("Gross margin")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(lambda x, pos: fmt_money(x))
        plt.legend(title="Category", loc="lower right")
        savefig("01_top_products_by_margin.png")

        scatter_df = product_summary.query("revenue > revenue.quantile(0.55)").copy()
        scatter_df["revenue_b"] = scatter_df["revenue"] / 1e9
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=scatter_df,
            x="units",
            y="revenue",
            hue="category",
            size="gross_margin_rate",
            sizes=(30, 300),
            alpha=0.75,
        )
        ax.set_title("Volume không tự động chuyển thành revenue/margin")
        ax.set_xlabel("Units sold")
        ax.set_ylabel("Revenue")
        ax.yaxis.set_major_formatter(lambda x, pos: fmt_money(x))
        for _, row in top_margin.head(5).iterrows():
            ax.text(row["units"], row["revenue"], row["product_name"], fontsize=8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig("02_units_vs_revenue_margin.png")
        """
    )
)


cells.append(md("## 4. Thời điểm vàng: category và top product bán mạnh khi nào?"))

cells.append(
    code(
        """
        category_month = (
            fact.groupby(["category", "month"], as_index=False)
            .agg(revenue=("line_revenue", "sum"), units=("quantity", "sum"), gross_margin=("gross_margin", "sum"))
        )
        category_month["avg_month_revenue"] = category_month.groupby("category")["revenue"].transform("mean")
        category_month["season_index"] = category_month["revenue"] / category_month["avg_month_revenue"]

        peak_category_month = (
            category_month.loc[category_month.groupby("category")["season_index"].idxmax()]
            .sort_values("season_index", ascending=False)
            .reset_index(drop=True)
        )

        display(
            view_table(
                peak_category_month[["category", "month", "revenue", "units", "gross_margin", "season_index"]],
                money_cols=["revenue", "gross_margin"],
                int_cols=["units"],
            )
        )

        plt.figure(figsize=(12, 4.8))
        heat = category_month.pivot(index="category", columns="month", values="season_index")
        ax = sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.4, cbar_kws={"label": "Revenue / monthly average"})
        ax.set_title("Seasonality index theo category và tháng")
        ax.set_xlabel("Month")
        ax.set_ylabel("")
        savefig("03_category_month_seasonality.png")
        """
    )
)


cells.append(
    code(
        """
        product_month = (
            fact.groupby(["product_id", "product_name", "category", "segment", "month"], as_index=False)
            .agg(revenue=("line_revenue", "sum"), units=("quantity", "sum"), gross_margin=("gross_margin", "sum"))
        )
        product_month["avg_month_revenue"] = product_month.groupby("product_id")["revenue"].transform("mean")
        product_month["season_index"] = product_month["revenue"] / product_month["avg_month_revenue"]

        peak_product_month = (
            product_month.loc[product_month.groupby("product_id")["season_index"].idxmax()]
            .rename(columns={"month": "peak_month", "season_index": "peak_season_index"})
            .merge(
                product_summary[["product_id", "revenue", "gross_margin", "promo_revenue_share", "return_rate_qty", "avg_rating", "n_reviews"]],
                on="product_id",
                how="left",
                suffixes=("_peak_month", "_total"),
            )
        )

        top_peak_products = (
            peak_product_month.sort_values("gross_margin_total", ascending=False)
            .head(15)
            [
                [
                    "product_name",
                    "category",
                    "segment",
                    "peak_month",
                    "peak_season_index",
                    "revenue_total",
                    "gross_margin_total",
                    "promo_revenue_share",
                    "return_rate_qty",
                    "avg_rating",
                    "n_reviews",
                ]
            ]
        )

        display(
            view_table(
                top_peak_products,
                money_cols=["revenue_total", "gross_margin_total"],
                pct_cols=["promo_revenue_share", "return_rate_qty"],
                int_cols=["n_reviews"],
            )
        )

        top_product_ids = product_summary.nlargest(8, "gross_margin")["product_id"]
        season_lines = product_month[product_month["product_id"].isin(top_product_ids)].copy()
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=season_lines, x="month", y="season_index", hue="product_name", marker="o")
        ax.axhline(1, color="black", linewidth=1, linestyle="--")
        ax.set_title("Mùa vụ của 8 sản phẩm đóng góp gross margin cao nhất")
        ax.set_xlabel("Month")
        ax.set_ylabel("Seasonality index")
        ax.set_xticks(range(1, 13))
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Product")
        savefig("04_top_product_seasonality.png")

        display(
            Markdown(
                "Khuyến nghị: planning promotion và tồn kho nên theo peak month của từng category/product, không dùng một lịch campaign chung cho toàn bộ danh mục."
            )
        )
        peak_product_month.to_csv(TABLE / "04_product_peak_month.csv", index=False)
        """
    )
)


cells.append(md("## 5. Kênh bán và mức phụ thuộc promotion"))

cells.append(
    code(
        """
        source_summary = (
            fact.groupby("order_source", as_index=False)
            .agg(
                orders=("order_id", "nunique"),
                units=("quantity", "sum"),
                revenue=("line_revenue", "sum"),
                gross_margin=("gross_margin", "sum"),
                promo_revenue=("promo_revenue", "sum"),
            )
        )
        source_summary["revenue_share"] = source_summary["revenue"] / source_summary["revenue"].sum()
        source_summary["gross_margin_rate"] = source_summary["gross_margin"] / source_summary["revenue"]
        source_summary["promo_revenue_share"] = source_summary["promo_revenue"] / source_summary["revenue"]
        source_summary = source_summary.sort_values("gross_margin", ascending=False)

        display(
            view_table(
                source_summary,
                money_cols=["revenue", "gross_margin", "promo_revenue"],
                pct_cols=["revenue_share", "gross_margin_rate", "promo_revenue_share"],
                int_cols=["orders", "units"],
            )
        )

        plt.figure(figsize=(10.5, 5))
        ax = sns.barplot(data=source_summary, x="order_source", y="gross_margin", hue="order_source", dodge=False)
        ax.set_title("Gross margin theo kênh đặt hàng")
        ax.set_xlabel("")
        ax.set_ylabel("Gross margin")
        ax.yaxis.set_major_formatter(lambda x, pos: fmt_money(x))
        plt.xticks(rotation=25, ha="right")
        plt.legend([], [], frameon=False)
        savefig("05_margin_by_order_source.png")

        source_category = (
            fact.groupby(["category", "order_source"], as_index=False)
            .agg(revenue=("line_revenue", "sum"))
        )
        source_category["category_revenue_share"] = source_category["revenue"] / source_category.groupby("category")["revenue"].transform("sum")
        source_pivot = source_category.pivot(index="category", columns="order_source", values="category_revenue_share").fillna(0)
        plt.figure(figsize=(11, 4.8))
        ax = sns.heatmap(source_pivot * 100, annot=True, fmt=".1f", cmap="Blues", linewidths=0.4, cbar_kws={"label": "% category revenue"})
        ax.set_title("Tỷ trọng revenue của từng category theo order_source")
        ax.set_xlabel("Order source")
        ax.set_ylabel("")
        savefig("06_category_source_mix.png")
        source_summary.to_csv(TABLE / "05_order_source_summary.csv", index=False)
        """
    )
)


cells.append(
    code(
        """
        min_revenue = product_summary["revenue"].quantile(0.75)
        min_units = product_summary["units"].quantile(0.75)

        promo_dependent = (
            product_summary.query("revenue >= @min_revenue and units >= @min_units")
            .sort_values(["promo_revenue_share", "revenue"], ascending=False)
            .head(15)
        )

        organic_winners = (
            product_summary.query("revenue >= @min_revenue and gross_margin_rate > 0")
            .sort_values(["promo_revenue_share", "gross_margin"], ascending=[True, False])
            .head(12)
        )

        display(Markdown("### Sản phẩm phụ thuộc promotion cao trong nhóm doanh thu/volume lớn"))
        display(
            view_table(
                promo_dependent[
                    [
                        "product_name",
                        "category",
                        "segment",
                        "units",
                        "revenue",
                        "gross_margin",
                        "gross_margin_rate",
                        "promo_revenue_share",
                        "discount_rate",
                        "return_rate_qty",
                    ]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["gross_margin_rate", "promo_revenue_share", "discount_rate", "return_rate_qty"],
                int_cols=["units"],
            )
        )

        display(Markdown("### Sản phẩm bán tốt không quá phụ thuộc promotion"))
        display(
            view_table(
                organic_winners[
                    [
                        "product_name",
                        "category",
                        "segment",
                        "units",
                        "revenue",
                        "gross_margin",
                        "gross_margin_rate",
                        "promo_revenue_share",
                        "return_rate_qty",
                    ]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["gross_margin_rate", "promo_revenue_share", "return_rate_qty"],
                int_cols=["units"],
            )
        )

        plot_df = product_summary.query("revenue >= @min_revenue").copy()
        plot_df["revenue_b"] = plot_df["revenue"] / 1e9
        plt.figure(figsize=(10.5, 6))
        ax = sns.scatterplot(
            data=plot_df,
            x="promo_revenue_share",
            y="gross_margin_rate",
            hue="category",
            size="revenue_b",
            sizes=(30, 420),
            alpha=0.78,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Promo dependency vs gross margin rate")
        ax.set_xlabel("Promotion revenue share")
        ax.set_ylabel("Gross margin rate")
        ax.xaxis.set_major_formatter(lambda x, pos: fmt_pct(x))
        ax.yaxis.set_major_formatter(lambda y, pos: fmt_pct(y))
        for _, row in promo_dependent.head(7).iterrows():
            ax.text(row["promo_revenue_share"], row["gross_margin_rate"], row["product_name"], fontsize=8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig("07_promo_dependency_margin.png")

        display(
            Markdown(
                f'''
                **Insight.** Có `{fmt_pct(fact['promo_used'].mean())}` dòng item dùng promotion, đóng góp `{fmt_pct(fact['promo_revenue'].sum() / fact['line_revenue'].sum())}` revenue.
                Nhóm promo-dependent cần được kiểm tra margin trước khi scale: sản phẩm có promo share cao nhưng margin thấp/âm nên dùng promotion để xả tồn hoặc acquisition, không nên mặc định là growth engine.
                '''
            )
        )
        promo_dependent.to_csv(TABLE / "06_promo_dependent_products.csv", index=False)
        organic_winners.to_csv(TABLE / "07_organic_winner_products.csv", index=False)
        """
    )
)


cells.append(
    code(
        """
        promo_fact = fact[fact["promo_id"].notna()].merge(
            promotions[["promo_id", "promo_name", "promo_type", "discount_value", "promo_channel", "applicable_category"]],
            on="promo_id",
            how="left",
        )
        promo_channel_summary = (
            promo_fact.groupby("promo_channel", dropna=False, as_index=False)
            .agg(
                promoted_lines=("line_id", "size"),
                units=("quantity", "sum"),
                revenue=("line_revenue", "sum"),
                gross_margin=("gross_margin", "sum"),
                avg_discount=("discount_amount", "mean"),
            )
            .sort_values("gross_margin", ascending=False)
        )
        promo_channel_summary["gross_margin_rate"] = promo_channel_summary["gross_margin"] / promo_channel_summary["revenue"]

        display(
            view_table(
                promo_channel_summary,
                money_cols=["revenue", "gross_margin", "avg_discount"],
                pct_cols=["gross_margin_rate"],
                int_cols=["promoted_lines", "units"],
            )
        )

        display(
            Markdown(
                "Lưu ý: bảng `promo_channel` chỉ phản ánh các dòng có `promo_id` chính. Nếu một dòng có `promo_id_2`, revenue không được double-count cho kênh thứ hai để tránh phóng đại doanh thu."
            )
        )
        promo_channel_summary.to_csv(TABLE / "08_promo_channel_summary.csv", index=False)
        """
    )
)


cells.append(md("## 6. Return và review: sản phẩm nào làm rò rỉ doanh thu?"))

cells.append(
    code(
        """
        returns_with_category = returns.merge(products[["product_id", "product_name", "category", "segment"]], on="product_id", how="left")
        reason_share = (
            returns_with_category["return_reason"]
            .value_counts(normalize=True)
            .rename_axis("return_reason")
            .reset_index(name="share")
        )
        display(view_table(reason_share, pct_cols=["share"]))

        reason_category = (
            returns_with_category.groupby(["category", "return_reason"], as_index=False)
            .agg(return_qty=("return_quantity", "sum"))
        )
        reason_category["share_within_category"] = reason_category["return_qty"] / reason_category.groupby("category")["return_qty"].transform("sum")
        reason_pivot = reason_category.pivot(index="category", columns="return_reason", values="share_within_category").fillna(0)

        plt.figure(figsize=(11, 4.8))
        ax = sns.heatmap(reason_pivot * 100, annot=True, fmt=".1f", cmap="OrRd", linewidths=0.4, cbar_kws={"label": "% returned quantity"})
        ax.set_title("Cơ cấu lý do hoàn hàng theo category")
        ax.set_xlabel("Return reason")
        ax.set_ylabel("")
        savefig("08_return_reason_by_category.png")

        min_reviews = 30
        quality_df = product_summary.query("n_reviews >= @min_reviews and units >= units.quantile(0.50)").copy()
        rating_return_corr = quality_df[["avg_rating", "return_rate_qty"]].corr().iloc[0, 1]

        risk_products = (
            product_summary.query("units >= @min_units and n_reviews >= @min_reviews")
            .assign(
                risk_score=lambda d: d["return_rate_qty"].fillna(0) * 0.65
                + d["refund_share"].fillna(0) * 0.25
                + (1 - d["avg_rating"].fillna(d["avg_rating"].median()) / 5) * 0.10
            )
            .sort_values(["return_rate_qty", "refund_share"], ascending=False)
            .head(15)
        )

        display(Markdown("### Sản phẩm volume lớn có return/refund risk cao"))
        display(
            view_table(
                risk_products[
                    [
                        "product_name",
                        "category",
                        "segment",
                        "units",
                        "revenue",
                        "gross_margin",
                        "return_rate_qty",
                        "refund_share",
                        "avg_rating",
                        "n_reviews",
                    ]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["return_rate_qty", "refund_share"],
                int_cols=["units", "n_reviews"],
            )
        )

        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=quality_df,
            x="avg_rating",
            y="return_rate_qty",
            hue="category",
            size="revenue",
            sizes=(25, 360),
            alpha=0.75,
        )
        ax.axhline(product_summary["return_rate_qty"].median(), color="black", linestyle="--", linewidth=1)
        ax.set_title("Average rating vs quantity return rate")
        ax.set_xlabel("Average rating")
        ax.set_ylabel("Return quantity / sold units")
        ax.yaxis.set_major_formatter(lambda y, pos: fmt_pct(y))
        for _, row in risk_products.head(7).iterrows():
            ax.text(row["avg_rating"], row["return_rate_qty"], row["product_name"], fontsize=8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig("09_rating_vs_return_rate.png")

        display(
            Markdown(
                f'''
                **Insight.** Lý do hoàn hàng lớn nhất là `{reason_share.iloc[0]['return_reason']}` với `{fmt_pct(reason_share.iloc[0]['share'])}` số dòng return.
                Tương quan product-level giữa rating và return rate trong nhóm có đủ review là `{rating_return_corr:.3f}`, nên rating không đủ thay thế return analysis.
                Nên xử lý return bằng reason cụ thể, đặc biệt size/defect/description, thay vì chỉ nhìn điểm review trung bình.
                '''
            )
        )
        risk_products.to_csv(TABLE / "09_return_risk_products.csv", index=False)
        reason_share.to_csv(TABLE / "10_return_reason_share.csv", index=False)
        """
    )
)


cells.append(md("## 7. Khu vực nào mua từng danh mục/sản phẩm?"))

cells.append(
    code(
        """
        fact_geo = fact.merge(geography[["zip", "city", "region", "district"]], on="zip", how="left")

        region_category = (
            fact_geo.groupby(["category", "region"], dropna=False, as_index=False)
            .agg(units=("quantity", "sum"), revenue=("line_revenue", "sum"), gross_margin=("gross_margin", "sum"))
        )
        region_category["category_region_share"] = region_category["revenue"] / region_category.groupby("category")["revenue"].transform("sum")

        top_region_by_category = (
            region_category.loc[region_category.groupby("category")["revenue"].idxmax()]
            .sort_values("category")
            .reset_index(drop=True)
        )
        display(
            view_table(
                top_region_by_category,
                money_cols=["revenue", "gross_margin"],
                pct_cols=["category_region_share"],
                int_cols=["units"],
            )
        )

        region_pivot = region_category.pivot(index="category", columns="region", values="category_region_share").fillna(0)
        plt.figure(figsize=(10, 4.8))
        ax = sns.heatmap(region_pivot * 100, annot=True, fmt=".1f", cmap="Greens", linewidths=0.4, cbar_kws={"label": "% category revenue"})
        ax.set_title("Demand concentration theo region")
        ax.set_xlabel("Region")
        ax.set_ylabel("")
        savefig("10_category_region_share.png")

        city_products = (
            fact_geo.groupby(["product_id", "product_name", "category", "city"], as_index=False)
            .agg(revenue=("line_revenue", "sum"), units=("quantity", "sum"))
        )
        city_products["product_city_share"] = city_products["revenue"] / city_products.groupby("product_id")["revenue"].transform("sum")
        top_city_for_margin_products = (
            city_products[city_products["product_id"].isin(top_margin["product_id"])]
            .sort_values(["product_id", "product_city_share"], ascending=[True, False])
            .groupby("product_id", as_index=False)
            .head(1)
            .merge(top_margin[["product_id", "gross_margin", "promo_revenue_share", "return_rate_qty"]], on="product_id", how="left")
            .sort_values("gross_margin", ascending=False)
        )
        display(Markdown("### Thành phố mạnh nhất của top product theo gross margin"))
        display(
            view_table(
                top_city_for_margin_products[
                    ["product_name", "category", "city", "revenue", "units", "product_city_share", "gross_margin", "promo_revenue_share", "return_rate_qty"]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["product_city_share", "promo_revenue_share", "return_rate_qty"],
                int_cols=["units"],
            )
        )

        display(
            Markdown(
                "Không có tồn kho theo region, nên phần location dùng để định hướng demand/campaign. Không nên kết luận tái phân bổ kho theo vùng nếu chưa có warehouse-level stock."
            )
        )
        region_category.to_csv(TABLE / "11_region_category_summary.csv", index=False)
        top_city_for_margin_products.to_csv(TABLE / "12_top_product_city_focus.csv", index=False)
        """
    )
)


cells.append(md("## 8. Tồn kho: có đủ hàng trong thời gian vàng không, và hàng nào tồn lâu?"))

cells.append(
    code(
        """
        inv_month = (
            inventory.groupby(["product_id", "month"], as_index=False)
            .agg(
                inv_units_sold=("units_sold", "sum"),
                avg_stock_on_hand=("stock_on_hand", "mean"),
                avg_units_received=("units_received", "mean"),
                avg_stockout_days=("stockout_days", "mean"),
                avg_fill_rate=("fill_rate", "mean"),
                avg_days_supply=("days_of_supply", "mean"),
                stockout_rate=("stockout_flag", "mean"),
                overstock_rate=("overstock_flag", "mean"),
                reorder_rate=("reorder_flag", "mean"),
                avg_sell_through=("sell_through_rate", "mean"),
            )
        )

        peak_for_inventory = peak_product_month[["product_id", "product_name", "category", "segment", "peak_month", "peak_season_index", "revenue_total", "gross_margin_total"]].copy()
        peak_inv = inv_month.rename(
            columns={
                "month": "peak_month",
                "avg_stockout_days": "peak_avg_stockout_days",
                "avg_fill_rate": "peak_avg_fill_rate",
                "avg_days_supply": "peak_avg_days_supply",
                "stockout_rate": "peak_stockout_rate",
                "overstock_rate": "peak_overstock_rate",
                "avg_sell_through": "peak_avg_sell_through",
                "avg_stock_on_hand": "peak_avg_stock_on_hand",
            }
        )
        readiness = peak_for_inventory.merge(
            peak_inv[
                [
                    "product_id",
                    "peak_month",
                    "peak_avg_stock_on_hand",
                    "peak_avg_stockout_days",
                    "peak_avg_fill_rate",
                    "peak_avg_days_supply",
                    "peak_stockout_rate",
                    "peak_overstock_rate",
                    "peak_avg_sell_through",
                ]
            ],
            on=["product_id", "peak_month"],
            how="left",
        )
        readiness["prev_month"] = np.where(readiness["peak_month"] == 1, 12, readiness["peak_month"] - 1)
        prev_inv = inv_month.rename(
            columns={
                "month": "prev_month",
                "avg_stockout_days": "prev_avg_stockout_days",
                "avg_fill_rate": "prev_avg_fill_rate",
                "avg_days_supply": "prev_avg_days_supply",
                "stockout_rate": "prev_stockout_rate",
                "overstock_rate": "prev_overstock_rate",
                "avg_sell_through": "prev_avg_sell_through",
                "avg_stock_on_hand": "prev_avg_stock_on_hand",
            }
        )
        readiness = readiness.merge(
            prev_inv[
                [
                    "product_id",
                    "prev_month",
                    "prev_avg_stock_on_hand",
                    "prev_avg_stockout_days",
                    "prev_avg_fill_rate",
                    "prev_avg_days_supply",
                    "prev_stockout_rate",
                    "prev_overstock_rate",
                    "prev_avg_sell_through",
                ]
            ],
            on=["product_id", "prev_month"],
            how="left",
        )
        readiness["inventory_risk_score"] = (
            readiness["peak_avg_stockout_days"].fillna(0)
            + readiness["prev_avg_stockout_days"].fillna(0)
            + (1 - readiness["peak_avg_fill_rate"].fillna(1)) * 30
            + (1 - readiness["prev_avg_fill_rate"].fillna(1)) * 30
        )

        top_margin_readiness = (
            readiness[readiness["product_id"].isin(top_margin["product_id"])]
            .sort_values("gross_margin_total", ascending=False)
        )
        display(Markdown("### Inventory readiness của top sản phẩm theo gross margin"))
        display(
            view_table(
                top_margin_readiness[
                    [
                        "product_name",
                        "category",
                        "peak_month",
                        "peak_season_index",
                        "gross_margin_total",
                        "prev_avg_stockout_days",
                        "prev_avg_fill_rate",
                        "peak_avg_stockout_days",
                        "peak_avg_fill_rate",
                        "peak_avg_days_supply",
                        "inventory_risk_score",
                    ]
                ],
                money_cols=["gross_margin_total"],
                pct_cols=["prev_avg_fill_rate", "peak_avg_fill_rate"],
            )
        )

        high_demand_stockout = (
            readiness.query("gross_margin_total >= gross_margin_total.quantile(0.75)")
            .sort_values("inventory_risk_score", ascending=False)
            .head(15)
        )
        display(Markdown("### High-demand products có rủi ro stockout quanh tháng peak"))
        display(
            view_table(
                high_demand_stockout[
                    [
                        "product_name",
                        "category",
                        "peak_month",
                        "peak_season_index",
                        "revenue_total",
                        "gross_margin_total",
                        "prev_avg_stockout_days",
                        "peak_avg_stockout_days",
                        "peak_avg_fill_rate",
                        "inventory_risk_score",
                    ]
                ],
                money_cols=["revenue_total", "gross_margin_total"],
                pct_cols=["peak_avg_fill_rate"],
            )
        )
        """
    )
)


cells.append(
    code(
        """
        inv_product = (
            inventory.groupby(["product_id", "product_name", "category", "segment"], as_index=False)
            .agg(
                avg_stock_on_hand=("stock_on_hand", "mean"),
                total_units_sold_inv=("units_sold", "sum"),
                avg_units_received=("units_received", "mean"),
                avg_stockout_days=("stockout_days", "mean"),
                stockout_rate=("stockout_flag", "mean"),
                overstock_rate=("overstock_flag", "mean"),
                reorder_rate=("reorder_flag", "mean"),
                avg_days_supply=("days_of_supply", "mean"),
                avg_sell_through=("sell_through_rate", "mean"),
                avg_fill_rate=("fill_rate", "mean"),
            )
        ).merge(
            product_summary[["product_id", "revenue", "gross_margin", "promo_revenue_share", "return_rate_qty"]],
            on="product_id",
            how="left",
        )

        inv_product["overstock_score"] = (
            inv_product["avg_stock_on_hand"]
            * inv_product["overstock_rate"].fillna(0)
            * (1 - inv_product["avg_sell_through"].fillna(0))
        )
        overstock_candidates = (
            inv_product.query("avg_stock_on_hand >= avg_stock_on_hand.quantile(0.70) and overstock_rate >= 0.70")
            .sort_values("overstock_score", ascending=False)
            .head(15)
        )

        display(Markdown("### Ứng viên tồn lâu/tồn nhiều"))
        display(
            view_table(
                overstock_candidates[
                    [
                        "product_name",
                        "category",
                        "segment",
                        "avg_stock_on_hand",
                        "avg_days_supply",
                        "avg_sell_through",
                        "overstock_rate",
                        "revenue",
                        "gross_margin",
                        "promo_revenue_share",
                        "return_rate_qty",
                        "overstock_score",
                    ]
                ],
                money_cols=["revenue", "gross_margin"],
                pct_cols=["avg_sell_through", "overstock_rate", "promo_revenue_share", "return_rate_qty"],
            )
        )

        plt.figure(figsize=(10.5, 6))
        plot_inv = inv_product.dropna(subset=["avg_days_supply", "avg_sell_through"]).copy()
        ax = sns.scatterplot(
            data=plot_inv,
            x="avg_days_supply",
            y="avg_sell_through",
            hue="category",
            size="avg_stock_on_hand",
            sizes=(20, 380),
            alpha=0.6,
        )
        ax.axhline(plot_inv["avg_sell_through"].median(), color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title("Inventory overstock map: days of supply cao nhưng sell-through thấp")
        ax.set_xlabel("Average days of supply (log scale)")
        ax.set_ylabel("Average sell-through rate")
        ax.yaxis.set_major_formatter(lambda y, pos: fmt_pct(y))
        for _, row in overstock_candidates.head(7).iterrows():
            ax.text(row["avg_days_supply"], row["avg_sell_through"], row["product_name"], fontsize=8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig("11_inventory_overstock_map.png")

        readiness.to_csv(TABLE / "13_inventory_peak_readiness.csv", index=False)
        overstock_candidates.to_csv(TABLE / "14_overstock_candidates.csv", index=False)
        inv_product.to_csv(TABLE / "15_inventory_product_summary.csv", index=False)

        display(
            Markdown(
                "Inventory insight dùng snapshot tháng nên phù hợp cho quyết định planning tháng. Nếu muốn quyết định replenishment theo ngày, cần thêm daily stock hoặc warehouse transaction log."
            )
        )
        """
    )
)


cells.append(md("## 9. Tổng hợp insight và hướng kể chuyện cho report Phase 2"))

cells.append(
    code(
        """
        top_cat = category_summary.sort_values("gross_margin", ascending=False).iloc[0]
        top_prod = top_margin.iloc[0]
        highest_promo = promo_dependent.iloc[0]
        riskiest_return = risk_products.iloc[0]
        top_overstock = overstock_candidates.iloc[0]
        peak_cat_text = ", ".join(
            f"{r.category}: tháng {int(r.month)} ({r.season_index:.2f}x)"
            for r in peak_category_month.itertuples(index=False)
        )

        synthesis = f'''
        ### Executive synthesis

        1. **Profit pool tập trung ở sản phẩm/danh mục khác với volume pool.** `{top_cat['category']}` là category dẫn đầu gross margin ({fmt_money(top_cat['gross_margin'])}), còn sản phẩm margin cao nhất là `{top_prod['product_name']}` ({fmt_money(top_prod['gross_margin'])}). Top sản phẩm theo units không nên mặc định là ưu tiên scale nếu margin thấp hoặc âm.

        2. **Seasonality đủ mạnh để lên lịch inventory/promotion theo category.** Peak month theo category: {peak_cat_text}. Đây là cơ sở để đặt hàng trước mùa cao điểm và kiểm soát discount theo mùa.

        3. **Promotion là đòn bẩy lớn nhưng cần guardrail margin.** Các dòng item có promotion đóng góp {fmt_pct(fact['promo_revenue'].sum() / fact['line_revenue'].sum())} revenue. `{highest_promo['product_name']}` thuộc nhóm volume/revenue lớn có promo revenue share {fmt_pct(highest_promo['promo_revenue_share'])}; nếu margin thấp, promotion nên được dùng có mục tiêu thay vì scale đại trà.

        4. **Return leakage nên xử lý theo nguyên nhân, không chỉ rating.** Lý do return lớn nhất là `{reason_share.iloc[0]['return_reason']}` ({fmt_pct(reason_share.iloc[0]['share'])}). Product risk cao nhất trong nhóm volume lớn là `{riskiest_return['product_name']}` với return rate {fmt_pct(riskiest_return['return_rate_qty'])} và refund share {fmt_pct(riskiest_return['refund_share'])}.

        5. **Demand theo vùng có thể guide campaign, nhưng chưa đủ để quyết định kho vùng.** `orders.zip` + `geography` cho thấy demand concentration theo region/city; tuy nhiên inventory không có dimension region/warehouse.

        6. **Inventory có hai bài toán song song: bảo vệ peak demand và giảm overstock.** Sản phẩm overstock nổi bật là `{top_overstock['product_name']}` với days of supply trung bình {top_overstock['avg_days_supply']:.1f} và sell-through {fmt_pct(top_overstock['avg_sell_through'])}. Nhóm high-demand stockout nên được ưu tiên replenishment trước peak month.

        ### Cách đưa vào report

        - Mở bằng nghịch lý: "best seller chưa chắc là profit driver".
        - Sau đó nối sang seasonality: profit driver cần được bảo vệ bằng inventory đúng tháng.
        - Tiếp theo là channel/promotion: scale channel nào và sản phẩm nào cần guardrail discount.
        - Kết thúc bằng operational actions: giảm return theo nguyên nhân, xử lý overstock, và tạo feature cho forecasting Phase 3.
        '''

        display(Markdown(synthesis))

        with open(WORK / "phase2_product_insights_summary.md", "w", encoding="utf-8") as f:
            f.write(synthesis.strip() + "\\n")

        print("Saved figures to:", FIG)
        print("Saved summary tables to:", TABLE)
        print("Saved markdown summary to:", WORK / "phase2_product_insights_summary.md")
        """
    )
)


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH}")
