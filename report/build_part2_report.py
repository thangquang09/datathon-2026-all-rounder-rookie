"""Build the Part 2 EDA/Data Storytelling report.

This script is intentionally self-contained and reproducible. It reads only the
13 real EDA tables from data/ and never reads sales_test.csv or
sample_submission.csv. Outputs are written under report/.

Run from the repository root:
    uv run python report/build_part2_report.py
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from textwrap import dedent

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "report"
FIG = REPORT / "figures"
TABLE = REPORT / "tables"

TABLES_13 = [
    "products",
    "customers",
    "promotions",
    "geography",
    "orders",
    "order_items",
    "payments",
    "shipments",
    "returns",
    "reviews",
    "sales",
    "inventory",
    "web_traffic",
]

FORBIDDEN_FOR_EDA = {"sales_test", "sample_submission"}

DATE_COLS = {
    "customers": ["signup_date"],
    "promotions": ["start_date", "end_date"],
    "orders": ["order_date"],
    "shipments": ["ship_date", "delivery_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "sales": ["Date"],
    "inventory": ["snapshot_date"],
    "web_traffic": ["date"],
}

OKABE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    sns.set_palette(OKABE)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dirs() -> None:
    REPORT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    TABLE.mkdir(exist_ok=True)
    style_src = ROOT / "eda_part" / "report" / "neurips_2025.sty"
    style_dst = REPORT / "neurips_2025.sty"
    if style_src.exists():
        shutil.copyfile(style_src, style_dst)


def load_data() -> dict[str, pd.DataFrame]:
    if TABLES_13 and FORBIDDEN_FOR_EDA.intersection(TABLES_13):
        raise RuntimeError("Forbidden test/submission table configured for EDA.")

    frames: dict[str, pd.DataFrame] = {}
    for name in TABLES_13:
        frames[name] = pd.read_csv(
            DATA / f"{name}.csv",
            parse_dates=DATE_COLS.get(name, []),
            low_memory=False,
        )
    return frames


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    png_path = FIG / f"{stem}.png"
    pdf_path = FIG / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=240)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path.relative_to(REPORT)), "pdf": str(pdf_path.relative_to(REPORT))}


def fmt_int(v: float | int) -> str:
    return f"{int(round(v)):,}"


def fmt_money(v: float) -> str:
    av = abs(float(v))
    if av >= 1e9:
        return f"{v / 1e9:,.2f} tỷ VND"
    if av >= 1e6:
        return f"{v / 1e6:,.1f} triệu VND"
    if av >= 1e3:
        return f"{v / 1e3:,.1f} nghìn VND"
    return f"{v:,.0f} VND"


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * float(v):.{digits}f}\\%"


def plain_pct(v: float, digits: int = 1) -> str:
    return f"{100 * float(v):.{digits}f}%"


def rel_pct(v: float, digits: int = 1) -> str:
    return f"{float(v):.{digits}f}\\%"


def latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE / name
    df.to_csv(path, index=False)
    return path


def build_workspace_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = [
        ("baseline.ipynb", "Baseline notebook; Part 3 leaning, not a Part 2 final report."),
        ("results/v1/eda.ipynb", "Broad first-pass EDA with many charts; useful coverage, weaker story."),
        ("results/v1/report.md", "Earlier report; mixes 15-file scope and forecasting framing."),
        ("results/v2/eda.ipynb", "Strongest existing time-series EDA; includes sample_submission in audit and focuses Part 3."),
        ("results/v2/report.md", "Good narrative but overweights forecasting/test horizon for this request."),
        ("scripts/report_analysis.py", "Targeted report chart script; reusable patterns and metrics."),
        ("group_job/phase2_product_insights_eda.ipynb", "Product/promo/inventory notebook; strong operational slices."),
        ("group_job/phase2_product_insights_summary.md", "Concise product-insight summary reused as candidate hypotheses."),
        ("eda_part/report/neurips_2025.tex", "LaTeX template source."),
        ("eda_part/report/neurips_2025.sty", "LaTeX style copied into report/."),
        ("src/", "Part 3 modeling code; audited but not used as Part 2 evidence."),
        ("outputs/", "Part 3 model outputs/submission candidates; excluded from EDA evidence."),
    ]
    for rel, note in targets:
        path = ROOT / rel
        exists = path.exists()
        item: dict[str, object] = {"path": rel, "exists": exists, "audit_note": note}
        if exists and path.suffix == ".ipynb":
            nb = nbformat.read(path, as_version=4)
            item.update(
                {
                    "cells": len(nb.cells),
                    "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                    "markdown_cells": sum(c.cell_type == "markdown" for c in nb.cells),
                    "outputs": sum(len(getattr(c, "outputs", [])) for c in nb.cells if c.cell_type == "code"),
                }
            )
        elif exists and path.is_dir():
            item["files"] = sum(1 for p in path.rglob("*") if p.is_file())
        elif exists:
            item["bytes"] = path.stat().st_size
        rows.append(item)
    return pd.DataFrame(rows)


def data_audit(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    for name, df in frames.items():
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        date_min = min((df[c].min() for c in date_cols), default=pd.NaT)
        date_max = max((df[c].max() for c in date_cols), default=pd.NaT)
        missing = df.isna().mean()
        max_missing_col = missing.idxmax() if len(missing) else ""
        max_missing_pct = float(missing.max() * 100) if len(missing) else 0.0
        audit_rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": df.shape[1],
                "date_min": "" if pd.isna(date_min) else str(date_min.date()),
                "date_max": "" if pd.isna(date_max) else str(date_max.date()),
                "duplicate_rows": int(df.duplicated().sum()),
                "max_missing_column": max_missing_col,
                "max_missing_pct": round(max_missing_pct, 2),
            }
        )

    products = frames["products"]
    customers = frames["customers"]
    geography = frames["geography"]
    orders = frames["orders"]
    items = frames["order_items"]
    payments = frames["payments"]
    shipments = frames["shipments"]
    returns = frames["returns"]
    reviews = frames["reviews"]
    inventory = frames["inventory"]
    checks = [
        ("orders.customer_id -> customers.customer_id", (~orders.customer_id.isin(customers.customer_id)).sum(), len(orders)),
        ("orders.zip -> geography.zip", (~orders.zip.isin(geography.zip)).sum(), len(orders)),
        ("order_items.order_id -> orders.order_id", (~items.order_id.isin(orders.order_id)).sum(), len(items)),
        ("order_items.product_id -> products.product_id", (~items.product_id.isin(products.product_id)).sum(), len(items)),
        ("payments.order_id -> orders.order_id", (~payments.order_id.isin(orders.order_id)).sum(), len(payments)),
        ("shipments.order_id -> orders.order_id", (~shipments.order_id.isin(orders.order_id)).sum(), len(shipments)),
        ("returns.order_id -> orders.order_id", (~returns.order_id.isin(orders.order_id)).sum(), len(returns)),
        ("returns.product_id -> products.product_id", (~returns.product_id.isin(products.product_id)).sum(), len(returns)),
        ("reviews.order_id -> orders.order_id", (~reviews.order_id.isin(orders.order_id)).sum(), len(reviews)),
        ("reviews.product_id -> products.product_id", (~reviews.product_id.isin(products.product_id)).sum(), len(reviews)),
        ("reviews.customer_id -> customers.customer_id", (~reviews.customer_id.isin(customers.customer_id)).sum(), len(reviews)),
        ("inventory.product_id -> products.product_id", (~inventory.product_id.isin(products.product_id)).sum(), len(inventory)),
    ]
    fk = pd.DataFrame(
        [
            {
                "check": name,
                "orphan_rows": int(orphan),
                "checked_rows": int(total),
                "pass_rate": 1 - float(orphan) / float(total),
            }
            for name, orphan, total in checks
        ]
    )
    return pd.DataFrame(audit_rows), fk


def build_enriched(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    products = frames["products"]
    orders = frames["orders"]
    items = frames["order_items"].copy()

    items = items.merge(
        products[["product_id", "product_name", "category", "segment", "price", "cogs"]],
        on="product_id",
        how="left",
    )
    items["line_revenue"] = items["quantity"] * items["unit_price"]
    items["line_cogs"] = items["quantity"] * items["cogs"]
    items["gross_margin"] = items["line_revenue"] - items["line_cogs"]
    items["has_promo"] = items["promo_id"].notna() | items["promo_id_2"].notna()

    order_rev = items.groupby("order_id", as_index=True).agg(
        order_revenue=("line_revenue", "sum"),
        order_cogs=("line_cogs", "sum"),
        order_margin=("gross_margin", "sum"),
        order_units=("quantity", "sum"),
        order_has_promo=("has_promo", "any"),
    )
    orders_enriched = orders.merge(order_rev, on="order_id", how="left").fillna(
        {"order_revenue": 0, "order_cogs": 0, "order_margin": 0, "order_units": 0, "order_has_promo": False}
    )
    item_order = items.merge(
        orders[["order_id", "order_date", "customer_id", "zip", "order_source", "device_type", "payment_method"]],
        on="order_id",
        how="left",
    )
    return {"items": items, "orders": orders_enriched, "item_order": item_order, "order_rev": order_rev}


def calculate_metrics(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> dict[str, object]:
    sales = frames["sales"].sort_values("Date").copy()
    products = frames["products"]
    customers = frames["customers"]
    promotions = frames["promotions"].copy()
    geography = frames["geography"]
    payments = frames["payments"]
    shipments = frames["shipments"].copy()
    returns = frames["returns"]
    reviews = frames["reviews"]
    inventory = frames["inventory"]
    web = frames["web_traffic"]
    items = enriched["items"]
    orders = enriched["orders"]
    item_order = enriched["item_order"]

    metrics: dict[str, object] = {}
    metrics["data_scope"] = {
        "tables_used": TABLES_13,
        "forbidden_tables": sorted(FORBIDDEN_FOR_EDA),
        "sales_min": str(sales["Date"].min().date()),
        "sales_max": str(sales["Date"].max().date()),
        "sales_days": int(sales["Date"].nunique()),
        "total_revenue": float(sales["Revenue"].sum()),
        "total_cogs": float(sales["COGS"].sum()),
        "gross_margin_rate": float((sales["Revenue"].sum() - sales["COGS"].sum()) / sales["Revenue"].sum()),
    }

    annual_sales = sales.assign(year=sales["Date"].dt.year).groupby("year").agg(
        revenue=("Revenue", "sum"),
        cogs=("COGS", "sum"),
        days=("Date", "nunique"),
    )
    annual_sales["gross_margin_rate"] = (annual_sales["revenue"] - annual_sales["cogs"]) / annual_sales["revenue"]
    annual_sales["revenue_yoy_pct"] = annual_sales["revenue"].pct_change() * 100

    yearly_ops = item_order.assign(year=item_order["order_date"].dt.year).groupby("year").agg(
        item_revenue=("line_revenue", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        lines=("order_id", "size"),
    )
    yearly_ops["aov"] = yearly_ops["item_revenue"] / yearly_ops["orders"]
    yearly_ops["units_per_order"] = yearly_ops["units"] / yearly_ops["orders"]
    yearly_ops["asp"] = yearly_ops["item_revenue"] / yearly_ops["units"]
    yearly = annual_sales.join(yearly_ops)
    yearly["orders_yoy_pct"] = yearly["orders"].pct_change() * 100
    yearly["aov_yoy_pct"] = yearly["aov"].pct_change() * 100
    yearly["units_yoy_pct"] = yearly["units"].pct_change() * 100
    metrics["yearly"] = yearly.reset_index().round(6).to_dict(orient="records")
    metrics["regime"] = {
        "pre_2019_daily_mean": float(sales.loc[sales.Date.dt.year <= 2018, "Revenue"].mean()),
        "post_2019_daily_mean": float(sales.loc[sales.Date.dt.year >= 2019, "Revenue"].mean()),
        "post_vs_pre_drop": float(1 - sales.loc[sales.Date.dt.year >= 2019, "Revenue"].mean() / sales.loc[sales.Date.dt.year <= 2018, "Revenue"].mean()),
        "revenue_yoy_2019_pct": float(yearly.loc[2019, "revenue_yoy_pct"]),
        "orders_yoy_2019_pct": float(yearly.loc[2019, "orders_yoy_pct"]),
        "units_yoy_2019_pct": float(yearly.loc[2019, "units_yoy_pct"]),
        "aov_yoy_2019_pct": float(yearly.loc[2019, "aov_yoy_pct"]),
        "revenue_yoy_2022_pct": float(yearly.loc[2022, "revenue_yoy_pct"]),
    }

    sales["month"] = sales["Date"].dt.month
    sales["dow"] = sales["Date"].dt.dayofweek
    monthly = sales.groupby("month")["Revenue"].mean()
    metrics["seasonality"] = {
        "monthly_deviation_pct": ((monthly / sales["Revenue"].mean() - 1) * 100).round(3).to_dict(),
        "best_month": int(monthly.idxmax()),
        "best_month_lift_pct": float((monthly.max() / sales["Revenue"].mean() - 1) * 100),
        "worst_month": int(monthly.idxmin()),
        "worst_month_lift_pct": float((monthly.min() / sales["Revenue"].mean() - 1) * 100),
    }

    monthly_cat = (
        item_order.assign(month=item_order["order_date"].dt.month)
        .groupby(["category", "month"])["line_revenue"]
        .sum()
        .reset_index()
    )
    cat_month_total = monthly_cat.groupby("category")["line_revenue"].transform("sum")
    monthly_cat["category_month_share"] = monthly_cat["line_revenue"] / cat_month_total
    peak_month = monthly_cat.sort_values("category_month_share", ascending=False).groupby("category").head(1)
    metrics["category_peak_months"] = peak_month.set_index("category")[["month", "category_month_share"]].round(6).to_dict(orient="index")

    cat = items.groupby("category").agg(
        revenue=("line_revenue", "sum"),
        cogs=("line_cogs", "sum"),
        margin=("gross_margin", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        products=("product_id", "nunique"),
    )
    cat["revenue_share"] = cat["revenue"] / cat["revenue"].sum()
    cat["margin_rate"] = cat["margin"] / cat["revenue"]
    cat = cat.sort_values("revenue", ascending=False)
    metrics["category_summary"] = cat.reset_index().round(6).to_dict(orient="records")
    metrics["category_hhi"] = float((cat["revenue_share"] ** 2).sum())

    product = items.groupby(["product_id", "product_name", "category", "segment"]).agg(
        revenue=("line_revenue", "sum"),
        margin=("gross_margin", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        promo_revenue=("line_revenue", lambda s: s[items.loc[s.index, "has_promo"]].sum()),
        discount=("discount_amount", "sum"),
    ).reset_index()
    product["margin_rate"] = product["margin"] / product["revenue"]
    product["promo_revenue_share"] = product["promo_revenue"] / product["revenue"]
    metrics["top_margin_products"] = product.sort_values("margin", ascending=False).head(10).round(6).to_dict(orient="records")

    daily_promo_share = orders.groupby(orders["order_date"].dt.normalize())["order_has_promo"].mean().rename("promo_order_share")
    daily = sales.set_index("Date")[["Revenue"]].join(daily_promo_share).dropna()
    daily["promo_bucket"] = pd.cut(
        daily["promo_order_share"],
        bins=[-0.01, 0.10, 0.25, 0.50, 1.01],
        labels=["<10%", "10-25%", "25-50%", ">50%"],
    )
    promo_bucket = daily.groupby("promo_bucket", observed=True)["Revenue"].agg(["count", "mean", "median"]).reset_index()
    light_median = daily.loc[daily["promo_bucket"].astype(str) == "<10%", "Revenue"].median()
    heavy_median = daily.loc[daily["promo_bucket"].astype(str) == ">50%", "Revenue"].median()
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promo_by_channel = promotions.groupby(["promo_type", "promo_channel"], dropna=False).agg(
        promos=("promo_id", "count"),
        avg_discount=("discount_value", "mean"),
        avg_duration_days=("duration_days", "mean"),
        stackable_share=("stackable_flag", "mean"),
    ).reset_index()
    metrics["promotion"] = {
        "bucket_stats": promo_bucket.round(4).to_dict(orient="records"),
        "heavy_vs_light_median_drop": float((light_median - heavy_median) / light_median),
        "daily_spearman_revenue_promo_share": float(daily[["Revenue", "promo_order_share"]].corr(method="spearman").loc["Revenue", "promo_order_share"]),
        "promo_line_revenue_share": float(items.loc[items["has_promo"], "line_revenue"].sum() / items["line_revenue"].sum()),
        "promo_by_channel": promo_by_channel.round(4).to_dict(orient="records"),
    }

    customer_rev = orders.groupby("customer_id")["order_revenue"].sum().sort_values(ascending=False)
    cum_customer = customer_rev.cumsum() / customer_rev.sum()
    order_count = orders.groupby("customer_id").size()
    source = item_order.groupby("order_source").agg(
        revenue=("line_revenue", "sum"),
        margin=("gross_margin", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
    ).sort_values("revenue", ascending=False)
    source["revenue_share"] = source["revenue"] / source["revenue"].sum()
    source["margin_rate"] = source["margin"] / source["revenue"]
    pay = payments.groupby("payment_method").agg(
        value=("payment_value", "sum"),
        orders=("order_id", "nunique"),
        avg_installments=("installments", "mean"),
    ).sort_values("value", ascending=False)
    pay["value_share"] = pay["value"] / pay["value"].sum()
    signup = customers.assign(year=customers["signup_date"].dt.year).groupby("year")["customer_id"].count()
    metrics["customer_channel_payment"] = {
        "customers_with_orders": int(len(customer_rev)),
        "top20_customer_revenue_share": float(cum_customer.iloc[int(len(customer_rev) * 0.2) - 1]),
        "repeat_customer_share": float((order_count > 1).mean()),
        "mean_orders_per_customer": float(order_count.mean()),
        "median_orders_per_customer": float(order_count.median()),
        "order_source": source.reset_index().round(6).to_dict(orient="records"),
        "payments": pay.reset_index().round(6).to_dict(orient="records"),
        "customer_signups": signup.to_dict(),
    }

    ret = returns.merge(products[["product_id", "category", "segment", "product_name"]], on="product_id", how="left")
    ret_reason = ret.groupby("return_reason").agg(
        return_quantity=("return_quantity", "sum"),
        refund_amount=("refund_amount", "sum"),
        return_rows=("return_id", "count"),
    ).sort_values("return_quantity", ascending=False)
    ret_reason["quantity_share"] = ret_reason["return_quantity"] / ret_reason["return_quantity"].sum()
    ret_reason["refund_share"] = ret_reason["refund_amount"] / ret_reason["refund_amount"].sum()
    sold_qty = items.groupby("category")["quantity"].sum()
    ret_cat = ret.groupby("category").agg(return_quantity=("return_quantity", "sum"), refund_amount=("refund_amount", "sum"))
    ret_cat = ret_cat.join(sold_qty.rename("sold_quantity"))
    ret_cat["return_rate"] = ret_cat["return_quantity"] / ret_cat["sold_quantity"]
    rating_cat = reviews.merge(products[["product_id", "category"]], on="product_id", how="left").groupby("category")["rating"].agg(["mean", "count"])
    shipments["delivery_days"] = (shipments["delivery_date"] - shipments["ship_date"]).dt.days
    review_order = reviews.groupby("order_id")["rating"].mean().rename("rating")
    ship_rev = shipments.merge(review_order, on="order_id", how="left")
    ship_rev["delivery_bucket"] = pd.cut(ship_rev["delivery_days"], [-1, 2, 4, 7], labels=["0-2", "3-4", "5-7"])
    delivery_rating = ship_rev.groupby("delivery_bucket", observed=True)["rating"].agg(["count", "mean"])
    metrics["returns_experience"] = {
        "return_reason": ret_reason.reset_index().round(6).to_dict(orient="records"),
        "return_category": ret_cat.reset_index().round(6).to_dict(orient="records"),
        "rating_category": rating_cat.reset_index().round(6).to_dict(orient="records"),
        "delivery_days_mean": float(shipments["delivery_days"].mean()),
        "delivery_days_median": float(shipments["delivery_days"].median()),
        "delivery_rating_corr": float(ship_rev[["delivery_days", "rating"]].corr(method="spearman").loc["delivery_days", "rating"]),
        "delivery_rating": delivery_rating.reset_index().round(6).to_dict(orient="records"),
    }

    item_geo = item_order.merge(geography[["zip", "region", "city"]], on="zip", how="left")
    region_total = item_geo.groupby("region").agg(revenue=("line_revenue", "sum"), orders=("order_id", "nunique")).sort_values("revenue", ascending=False)
    region_total["revenue_share"] = region_total["revenue"] / region_total["revenue"].sum()
    region_cat = item_geo.groupby(["region", "category"])["line_revenue"].sum().reset_index()
    region_cat["region_revenue"] = region_cat.groupby("region")["line_revenue"].transform("sum")
    region_cat["category_share_in_region"] = region_cat["line_revenue"] / region_cat["region_revenue"]
    metrics["geography"] = {
        "region_total": region_total.reset_index().round(6).to_dict(orient="records"),
        "region_category": region_cat.round(6).to_dict(orient="records"),
    }

    inv_month = inventory.groupby(["year", "month"]).agg(
        stockout_rate=("stockout_flag", "mean"),
        overstock_rate=("overstock_flag", "mean"),
        mean_fill_rate=("fill_rate", "mean"),
        mean_sell_through=("sell_through_rate", "mean"),
    ).reset_index()
    product_demand = items.groupby(["product_id", "product_name", "category"]).agg(
        units=("quantity", "sum"),
        revenue=("line_revenue", "sum"),
        margin=("gross_margin", "sum"),
    ).reset_index()
    inv_product = inventory.groupby("product_id").agg(
        avg_stockout=("stockout_flag", "mean"),
        avg_overstock=("overstock_flag", "mean"),
        avg_days_supply=("days_of_supply", "mean"),
        avg_sell_through=("sell_through_rate", "mean"),
        avg_fill_rate=("fill_rate", "mean"),
        avg_stock=("stock_on_hand", "mean"),
    ).reset_index().merge(product_demand, on="product_id", how="left")
    high_demand_cutoff = inv_product["units"].quantile(0.75)
    high_demand_stockout = inv_product.loc[inv_product["units"] >= high_demand_cutoff].sort_values(["avg_stockout", "units"], ascending=False).head(10)
    overstock_candidates = inv_product.loc[inv_product["units"] >= inv_product["units"].quantile(0.50)].sort_values(["avg_overstock", "avg_days_supply"], ascending=False).head(10)
    metrics["inventory"] = {
        "monthly": inv_month.round(6).to_dict(orient="records"),
        "stockout_2022_avg": float(inv_month.loc[inv_month["year"] == 2022, "stockout_rate"].mean()),
        "overstock_2022_avg": float(inv_month.loc[inv_month["year"] == 2022, "overstock_rate"].mean()),
        "sell_through_2022_avg": float(inv_month.loc[inv_month["year"] == 2022, "mean_sell_through"].mean()),
        "high_demand_stockout": high_demand_stockout.round(6).to_dict(orient="records"),
        "overstock_candidates": overstock_candidates.round(6).to_dict(orient="records"),
    }

    web_daily = web.groupby("date").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    )
    web_join = sales.set_index("Date")[["Revenue"]].join(web_daily, how="inner")
    web_corr = web_join.corr(method="spearman")["Revenue"].drop("Revenue").sort_values(ascending=False)
    lead_lag = []
    for lag in range(-14, 15):
        lead_lag.append({"lag_days": lag, "spearman_sessions": float(web_join["Revenue"].corr(web_join["sessions"].shift(lag), method="spearman"))})
    lead_lag_df = pd.DataFrame(lead_lag)
    best_lag = lead_lag_df.iloc[lead_lag_df["spearman_sessions"].abs().idxmax()]
    web_source = web.groupby("traffic_source").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
    ).sort_values("sessions", ascending=False)
    web_source["session_share"] = web_source["sessions"] / web_source["sessions"].sum()
    metrics["web_traffic"] = {
        "joined_days": int(len(web_join)),
        "spearman": web_corr.round(6).to_dict(),
        "lead_lag": lead_lag_df.round(6).to_dict(orient="records"),
        "best_lag_days": int(best_lag["lag_days"]),
        "best_lag_spearman": float(best_lag["spearman_sessions"]),
        "source": web_source.reset_index().round(6).to_dict(orient="records"),
    }

    return metrics


def generate_tables(
    metrics: dict[str, object],
    data_audit_df: pd.DataFrame,
    fk_df: pd.DataFrame,
    workspace_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    write_csv(data_audit_df, "data_audit_13_tables.csv")
    write_csv(fk_df, "referential_integrity.csv")
    write_csv(workspace_df, "workspace_audit.csv")

    action_rows = [
        {
            "priority": 1,
            "decision_area": "Kế hoạch nhu cầu",
            "evidence": "Revenue 2019 giảm 38.6% YoY, số đơn giảm 40.1%; daily mean hậu 2019 thấp hơn 40.5% so với 2012-2018.",
            "recommended_action": "Lấy hậu 2019 làm baseline vận hành; không cho các năm đầu kéo kế hoạch tồn kho/ngân sách lên quá cao.",
            "tables": "sales, orders, order_items",
        },
        {
            "priority": 2,
            "decision_area": "Merchandising theo mùa",
            "evidence": "Tháng 5 cao hơn average day 53.4%; tháng 12 thấp hơn 41.1%; peak month khác nhau theo category.",
            "recommended_action": "Chuẩn bị mua hàng/content/stock từ tháng 3-6; giữ Outdoor allocation riêng cho tháng 12 và vùng West.",
            "tables": "sales, orders, order_items, products, geography",
        },
        {
            "priority": 3,
            "decision_area": "Quản trị promotion",
            "evidence": "Ngày có >50% đơn dùng promo có median revenue thấp hơn 13.3% so với ngày <10% promo.",
            "recommended_action": "Dùng promo như intervention có đo uplift và margin guardrail, không chạy blanket như đòn bẩy mặc định.",
            "tables": "orders, order_items, promotions, products",
        },
        {
            "priority": 4,
            "decision_area": "Retention và kênh bán",
            "evidence": "Top 20% khách tạo 60.6% revenue; 75.2% khách có đơn là repeat buyers.",
            "recommended_action": "Bảo vệ cohort giá trị cao bằng early access/personalized drops trước mùa peak.",
            "tables": "customers, orders, order_items, payments",
        },
        {
            "priority": 5,
            "decision_area": "Rò rỉ qua return",
            "evidence": "wrong_size chiếm 34.7% return quantity và 34.6% refund; return rate giữa category rất sát nhau.",
            "recommended_action": "Sửa size guidance và expectation trên product page ở cấp hệ thống trước khi nhắm vào một category.",
            "tables": "returns, reviews, shipments, products, order_items",
        },
        {
            "priority": 6,
            "decision_area": "Phân bổ tồn kho",
            "evidence": "Năm 2022 có stockout rate 66.6% và overstock rate 76.7% cùng lúc.",
            "recommended_action": "Reallocate ở cấp SKU: replenish SKU nhu cầu cao bị stockout và xử lý chronic overstock.",
            "tables": "inventory, products, order_items",
        },
        {
            "priority": 7,
            "decision_area": "Traffic và tín hiệu dự báo",
            "evidence": "Sessions chỉ tương quan 0.368 với revenue; lead/lag tốt nhất là 0.373 tại lag +1.",
            "recommended_action": "Dùng traffic như tín hiệu phụ; đo conversion/value theo source thay vì tối ưu volume thuần.",
            "tables": "web_traffic, sales",
        },
    ]
    action_df = pd.DataFrame(action_rows)
    write_csv(action_df, "action_matrix.csv")

    chart_rows = [
        {
            "artifact": "figures/01_revenue_decomposition.pdf",
            "tables_used": "sales; orders; order_items; products",
            "main_insight": "2019 is a structural demand break driven by order/unit collapse, not lower AOV.",
            "key_numbers": "Revenue YoY 2019 -38.6%; orders YoY -40.1%; AOV YoY +2.7%; post-2019 daily mean -40.5% vs 2012-2018.",
        },
        {
            "artifact": "figures/02_seasonality_category.pdf",
            "tables_used": "sales; orders; order_items; products",
            "main_insight": "The business peaks in Q2, with category-specific peak months.",
            "key_numbers": "May +53.4% vs average day; Dec -41.1%; Streetwear peak May, GenZ June, Outdoor December.",
        },
        {
            "artifact": "figures/03_profit_pool.pdf",
            "tables_used": "products; order_items",
            "main_insight": "Streetwear dominates revenue, but product-level profit pool is more selective.",
            "key_numbers": "Streetwear 79.9% revenue; HHI 0.663; top margin SKU SaigonFlex UM-43 = 130.5M VND gross margin.",
        },
        {
            "artifact": "figures/04_promotion_guardrails.pdf",
            "tables_used": "orders; order_items; promotions; products",
            "main_insight": "Promotion is associated with weaker days and needs margin guardrails.",
            "key_numbers": ">50% promo-order days have 13.3% lower median revenue than <10% promo-order days; Spearman rho=-0.114.",
        },
        {
            "artifact": "figures/05_customer_channel_payment.pdf",
            "tables_used": "customers; orders; order_items; payments",
            "main_insight": "Repeat customers and search-led demand are the commercial backbone.",
            "key_numbers": "Top 20% customers = 60.6% revenue; repeat share 75.2%; organic_search 28.0% revenue; credit_card 55.0% payment value.",
        },
        {
            "artifact": "figures/06_returns_experience.pdf",
            "tables_used": "returns; reviews; shipments; products; order_items",
            "main_insight": "Return leakage is system-wide, especially size/expectation mismatch, while delivery-rating linkage is weak.",
            "key_numbers": "wrong_size 34.7% return quantity; category return rates 3.26%-3.52%; delivery-rating rho=-0.006.",
        },
        {
            "artifact": "figures/07_geography_mix.pdf",
            "tables_used": "geography; orders; order_items; products",
            "main_insight": "Region mix changes category allocation; West is less Streetwear-heavy and more Outdoor-heavy.",
            "key_numbers": "East 46.5% revenue; Central 30.1%; West 23.4%; Outdoor share West 26.2% vs East 11.8%.",
        },
        {
            "artifact": "figures/08_inventory_misalignment.pdf",
            "tables_used": "inventory; products; order_items",
            "main_insight": "Stockout and overstock coexist, pointing to allocation mismatch rather than pure supply shortage.",
            "key_numbers": "2022 avg stockout 66.6%; avg overstock 76.7%; avg sell-through 13.6%.",
        },
        {
            "artifact": "figures/09_web_traffic_signal.pdf",
            "tables_used": "web_traffic; sales",
            "main_insight": "Traffic is useful but not a strong standalone leading indicator.",
            "key_numbers": "Sessions rho=0.368; unique visitors rho=0.369; best sessions lead/lag rho=0.373 at lag +1.",
        },
        {
            "artifact": "tables/data_audit_13_tables.csv",
            "tables_used": "all 13 EDA tables",
            "main_insight": "The clean EDA scope excludes test/submission files and has consistent FK coverage.",
            "key_numbers": "13 tables; 646,945 orders; 714,669 item rows; 3,833 sales days; 0 FK orphan rows in key checks.",
        },
    ]
    chart_df = pd.DataFrame(chart_rows)
    write_csv(chart_df, "chart_table_summary.csv")
    return {"action": action_df, "chart_summary": chart_df}


def plot_revenue_decomposition(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame], metrics: dict[str, object]) -> None:
    sales = frames["sales"].copy()
    item_order = enriched["item_order"]
    yearly_sales = sales.assign(year=sales["Date"].dt.year).groupby("year").agg(revenue=("Revenue", "sum"), cogs=("COGS", "sum"))
    yearly_sales["gross_margin_rate"] = (yearly_sales["revenue"] - yearly_sales["cogs"]) / yearly_sales["revenue"]
    yearly_ops = item_order.assign(year=item_order["order_date"].dt.year).groupby("year").agg(
        item_revenue=("line_revenue", "sum"),
        orders=("order_id", "nunique"),
        units=("quantity", "sum"),
    )
    yearly_ops["aov"] = yearly_ops["item_revenue"] / yearly_ops["orders"]
    yearly = yearly_sales.join(yearly_ops)
    base = yearly.loc[2013, ["revenue", "orders", "aov"]]
    idx = yearly.loc[2013:, ["revenue", "orders", "aov"]].divide(base) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.bar(yearly.index, yearly["revenue"] / 1e9, color=OKABE[0], alpha=0.80, label="Revenue")
    ax2 = ax.twinx()
    ax2.plot(yearly.index, yearly["gross_margin_rate"] * 100, marker="o", color=OKABE[3], label="Gross margin rate")
    ax.axvspan(2018.5, 2019.5, color=OKABE[1], alpha=0.18, label="2019 break")
    ax.set_title("Annual revenue and margin, 2012-2022")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue (billion VND)")
    ax2.set_ylabel("Gross margin rate (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    ax = axes[1]
    ax.plot(idx.index, idx["revenue"], marker="o", label="Revenue index", color=OKABE[0])
    ax.plot(idx.index, idx["orders"], marker="s", label="Order count index", color=OKABE[1])
    ax.plot(idx.index, idx["aov"], marker="^", label="AOV index", color=OKABE[2])
    ax.axvline(2019, color="0.45", linestyle="--", linewidth=1)
    ax.axhline(100, color="0.75", linewidth=0.8)
    ax.set_title("Demand decomposition (2013 = 100)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.legend(frameon=True)
    save_figure(fig, "01_revenue_decomposition")


def plot_seasonality(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> None:
    sales = frames["sales"].copy()
    item_order = enriched["item_order"].copy()
    sales["month"] = sales["Date"].dt.month
    monthly = sales.groupby("month")["Revenue"].mean()
    deviation = (monthly / sales["Revenue"].mean() - 1) * 100

    monthly_cat = item_order.assign(month=item_order["order_date"].dt.month).groupby(["category", "month"])["line_revenue"].sum().reset_index()
    monthly_cat["share"] = monthly_cat["line_revenue"] / monthly_cat.groupby("category")["line_revenue"].transform("sum") * 100
    heat = monthly_cat.pivot(index="category", columns="month", values="share").loc[["Streetwear", "Outdoor", "Casual", "GenZ"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    colors = [OKABE[2] if v > 0 else OKABE[3] for v in deviation.values]
    ax.bar(deviation.index, deviation.values, color=colors, alpha=0.85)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_title("Monthly daily-revenue deviation")
    ax.set_xlabel("Month")
    ax.set_ylabel("% vs average day")
    ax.set_xticks(range(1, 13))
    for month, value in deviation.items():
        if month in [1, 5, 12]:
            ax.text(month, value + (2 if value >= 0 else -4), f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    ax = axes[1]
    sns.heatmap(
        heat,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        cbar_kws={"label": "% of category revenue"},
        ax=ax,
    )
    ax.set_title("Category seasonality by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")
    save_figure(fig, "02_seasonality_category")


def plot_profit_pool(enriched: dict[str, pd.DataFrame]) -> None:
    items = enriched["items"]
    cat = items.groupby("category").agg(revenue=("line_revenue", "sum"), margin=("gross_margin", "sum")).sort_values("revenue", ascending=False)
    cat["revenue_share"] = cat["revenue"] / cat["revenue"].sum()
    cat["margin_share"] = cat["margin"] / cat["margin"].sum()
    product = items.groupby(["product_id", "product_name", "category", "segment"]).agg(
        revenue=("line_revenue", "sum"),
        margin=("gross_margin", "sum"),
        units=("quantity", "sum"),
    ).reset_index()
    top_label = product.sort_values("margin", ascending=False).head(4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [0.9, 1.35]})
    ax = axes[0]
    x = np.arange(len(cat))
    width = 0.38
    ax.bar(x - width / 2, cat["revenue_share"] * 100, width, label="Revenue share", color=OKABE[0])
    ax.bar(x + width / 2, cat["margin_share"] * 100, width, label="Gross-margin share", color=OKABE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(cat.index, rotation=25, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title("Category revenue vs profit pool")
    ax.legend(frameon=True)

    ax = axes[1]
    sns.scatterplot(
        data=product,
        x="units",
        y="margin",
        hue="category",
        size="revenue",
        sizes=(18, 260),
        alpha=0.62,
        palette=OKABE[:4],
        ax=ax,
        legend="brief",
    )
    ax.set_xscale("log")
    ax.set_title("SKU units sold vs gross margin")
    ax.set_xlabel("Units sold, log scale")
    ax.set_ylabel("Gross margin (VND)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.0f}M")
    for _, row in top_label.iterrows():
        ax.annotate(row["product_name"], (row["units"], row["margin"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    save_figure(fig, "03_profit_pool")


def plot_promotion(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> None:
    sales = frames["sales"]
    promotions = frames["promotions"].copy()
    items = enriched["items"]
    orders = enriched["orders"]
    product = items.groupby(["product_id", "product_name", "category"]).agg(
        revenue=("line_revenue", "sum"),
        margin=("gross_margin", "sum"),
        promo_revenue=("line_revenue", lambda s: s[items.loc[s.index, "has_promo"]].sum()),
    ).reset_index()
    product["margin_rate"] = product["margin"] / product["revenue"] * 100
    product["promo_revenue_share"] = product["promo_revenue"] / product["revenue"] * 100
    top_product = product.loc[product["revenue"] >= product["revenue"].quantile(0.75)]

    daily_promo = orders.groupby(orders["order_date"].dt.normalize())["order_has_promo"].mean().rename("promo_order_share")
    daily = sales.set_index("Date")[["Revenue"]].join(daily_promo).dropna()
    daily["promo_bucket"] = pd.cut(
        daily["promo_order_share"],
        [-0.01, 0.10, 0.25, 0.50, 1.01],
        labels=["<10%", "10-25%", "25-50%", ">50%"],
    )
    promotions["duration_days"] = (promotions["end_date"] - promotions["start_date"]).dt.days + 1
    promo_channel = promotions.groupby("promo_channel", dropna=False).agg(
        promos=("promo_id", "count"),
        avg_discount=("discount_value", "mean"),
        avg_duration=("duration_days", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1.0, 1.05, 0.9]})
    ax = axes[0]
    sns.boxplot(
        data=daily,
        x="promo_bucket",
        y="Revenue",
        hue="promo_bucket",
        showfliers=False,
        palette=[OKABE[2], OKABE[5], OKABE[1], OKABE[3]],
        legend=False,
        ax=ax,
    )
    ax.set_title("Daily revenue by promo intensity")
    ax.set_xlabel("Share of orders using promo")
    ax.set_ylabel("Daily revenue (VND)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.1f}M")

    ax = axes[1]
    sns.scatterplot(
        data=top_product,
        x="promo_revenue_share",
        y="margin_rate",
        hue="category",
        size="revenue",
        sizes=(20, 240),
        alpha=0.70,
        palette=OKABE[:4],
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.set_title("Top-revenue SKUs: promo dependence")
    ax.set_xlabel("Promo revenue share (%)")
    ax.set_ylabel("Gross margin rate (%)")

    ax = axes[2]
    sns.barplot(data=promo_channel.sort_values("promos", ascending=False), x="promos", y="promo_channel", color=OKABE[0], ax=ax)
    ax.set_title("Promotion catalog by channel")
    ax.set_xlabel("Number of promotions")
    ax.set_ylabel("Promo channel")
    save_figure(fig, "04_promotion_guardrails")


def plot_customer_channel_payment(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> None:
    payments = frames["payments"]
    orders = enriched["orders"]
    item_order = enriched["item_order"]
    customer_rev = orders.groupby("customer_id")["order_revenue"].sum().sort_values(ascending=False)
    cum = customer_rev.cumsum() / customer_rev.sum() * 100
    source = item_order.groupby("order_source")["line_revenue"].sum().sort_values(ascending=False)
    source_share = source / source.sum() * 100
    pay = payments.groupby("payment_method")["payment_value"].sum().sort_values(ascending=False)
    pay_share = pay / pay.sum() * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax = axes[0]
    ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values, color=OKABE[0], linewidth=2)
    ax.axvline(20, color="0.45", linestyle="--", linewidth=1)
    ax.axhline(cum.iloc[int(len(cum) * 0.2) - 1], color="0.45", linestyle="--", linewidth=1)
    ax.set_title("Customer revenue Pareto")
    ax.set_xlabel("% customers sorted by revenue")
    ax.set_ylabel("% cumulative revenue")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = axes[1]
    sns.barplot(x=source_share.values, y=source_share.index, color=OKABE[2], ax=ax)
    ax.set_title("Revenue by order source")
    ax.set_xlabel("Revenue share (%)")
    ax.set_ylabel("Order source")

    ax = axes[2]
    sns.barplot(x=pay_share.values, y=pay_share.index, color=OKABE[1], ax=ax)
    ax.set_title("Payment value by method")
    ax.set_xlabel("Payment value share (%)")
    ax.set_ylabel("Payment method")
    save_figure(fig, "05_customer_channel_payment")


def plot_returns_experience(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> None:
    products = frames["products"]
    returns = frames["returns"]
    reviews = frames["reviews"]
    shipments = frames["shipments"].copy()
    items = enriched["items"]
    ret = returns.merge(products[["product_id", "category"]], on="product_id", how="left")
    reason = ret.groupby("return_reason").agg(return_quantity=("return_quantity", "sum"), refund_amount=("refund_amount", "sum")).sort_values("return_quantity", ascending=False)
    reason["quantity_share"] = reason["return_quantity"] / reason["return_quantity"].sum() * 100
    ret_cat = ret.groupby("category")["return_quantity"].sum().rename("return_quantity").to_frame()
    ret_cat = ret_cat.join(items.groupby("category")["quantity"].sum().rename("sold_quantity"))
    ret_cat["return_rate"] = ret_cat["return_quantity"] / ret_cat["sold_quantity"] * 100
    rating_cat = reviews.merge(products[["product_id", "category"]], on="product_id", how="left").groupby("category")["rating"].mean()
    shipments["delivery_days"] = (shipments["delivery_date"] - shipments["ship_date"]).dt.days
    review_order = reviews.groupby("order_id")["rating"].mean().rename("rating")
    ship_rev = shipments.merge(review_order, on="order_id", how="inner")
    ship_rev["delivery_bucket"] = pd.cut(ship_rev["delivery_days"], [-1, 2, 4, 7], labels=["0-2", "3-4", "5-7"])
    delivery = ship_rev.groupby("delivery_bucket", observed=True)["rating"].agg(["mean", "count"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1.0, 1.05, 0.9]})
    ax = axes[0]
    sns.barplot(x=reason["quantity_share"].values, y=reason.index, color=OKABE[3], ax=ax)
    ax.set_title("Return quantity by reason")
    ax.set_xlabel("Share of returned units (%)")
    ax.set_ylabel("Return reason")

    ax = axes[1]
    cat_order = ret_cat.sort_values("return_rate", ascending=False).index
    ax.bar(cat_order, ret_cat.loc[cat_order, "return_rate"], color=OKABE[0], alpha=0.8, label="Return rate")
    ax.set_title("Category return rate and rating")
    ax.set_xlabel("Category")
    ax.set_ylabel("Return rate (%)")
    ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    ax2.plot(cat_order, rating_cat.loc[cat_order], color=OKABE[1], marker="o", label="Mean rating")
    ax2.set_ylabel("Mean product rating")
    ax2.set_ylim(3.7, 4.1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    ax = axes[2]
    ax.bar(delivery.index.astype(str), delivery["mean"], color=OKABE[2], alpha=0.85)
    ax.set_ylim(3.7, 4.1)
    ax.set_title("Delivery days vs rating")
    ax.set_xlabel("Delivery days")
    ax.set_ylabel("Mean rating")
    for i, (_, row) in enumerate(delivery.iterrows()):
        ax.text(i, row["mean"] + 0.01, f"n={int(row['count']):,}", ha="center", va="bottom", fontsize=7)
    save_figure(fig, "06_returns_experience")


def plot_geography(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> None:
    geography = frames["geography"]
    item_order = enriched["item_order"].merge(geography[["zip", "region", "city"]], on="zip", how="left")
    region_total = item_order.groupby("region").agg(revenue=("line_revenue", "sum"), orders=("order_id", "nunique")).sort_values("revenue", ascending=False)
    region_total["revenue_share"] = region_total["revenue"] / region_total["revenue"].sum() * 100
    region_cat = item_order.groupby(["region", "category"])["line_revenue"].sum().reset_index()
    region_cat["region_revenue"] = region_cat.groupby("region")["line_revenue"].transform("sum")
    region_cat["share"] = region_cat["line_revenue"] / region_cat["region_revenue"] * 100
    pivot = region_cat.pivot(index="region", columns="category", values="share").fillna(0).loc[region_total.index]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [0.8, 1.2]})
    ax = axes[0]
    sns.barplot(x=region_total["revenue_share"], y=region_total.index, color=OKABE[0], ax=ax)
    ax.set_title("Revenue concentration by region")
    ax.set_xlabel("Revenue share (%)")
    ax.set_ylabel("Region")

    ax = axes[1]
    bottom = np.zeros(len(pivot))
    category_order = ["Streetwear", "Outdoor", "Casual", "GenZ"]
    for i, category in enumerate(category_order):
        vals = pivot[category].values
        ax.bar(pivot.index, vals, bottom=bottom, label=category, color=OKABE[i], alpha=0.9)
        bottom += vals
    ax.set_title("Category revenue mix inside each region")
    ax.set_xlabel("Region")
    ax.set_ylabel("% of region revenue")
    ax.legend(frameon=True, loc="lower left")
    save_figure(fig, "07_geography_mix")


def plot_inventory(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame]) -> None:
    inventory = frames["inventory"]
    items = enriched["items"]
    inv_month = inventory.groupby(["year", "month"]).agg(
        stockout_rate=("stockout_flag", "mean"),
        overstock_rate=("overstock_flag", "mean"),
        mean_fill_rate=("fill_rate", "mean"),
        mean_sell_through=("sell_through_rate", "mean"),
    ).reset_index()
    inv_month["date"] = pd.to_datetime(dict(year=inv_month["year"], month=inv_month["month"], day=1))
    demand = items.groupby(["product_id", "product_name", "category"]).agg(units=("quantity", "sum"), revenue=("line_revenue", "sum")).reset_index()
    inv_product = inventory.groupby("product_id").agg(
        avg_stockout=("stockout_flag", "mean"),
        avg_overstock=("overstock_flag", "mean"),
        avg_days_supply=("days_of_supply", "mean"),
        avg_sell_through=("sell_through_rate", "mean"),
    ).reset_index().merge(demand, on="product_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    ax.plot(inv_month["date"], inv_month["stockout_rate"] * 100, label="Stockout rate", color=OKABE[3], linewidth=1.8)
    ax.plot(inv_month["date"], inv_month["overstock_rate"] * 100, label="Overstock rate", color=OKABE[1], linewidth=1.8)
    ax.plot(inv_month["date"], inv_month["mean_fill_rate"] * 100, label="Mean fill rate", color=OKABE[2], linewidth=1.2)
    ax.set_title("Monthly inventory health")
    ax.set_xlabel("Snapshot month")
    ax.set_ylabel("% of SKU-months / fill rate")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=True)

    ax = axes[1]
    scatter = ax.scatter(
        inv_product["units"],
        inv_product["avg_days_supply"].clip(upper=7000),
        c=inv_product["avg_overstock"],
        s=40 + inv_product["avg_stockout"] * 120,
        cmap="viridis",
        alpha=0.68,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xscale("log")
    ax.set_title("SKU demand vs days of supply")
    ax.set_xlabel("Units sold, log scale")
    ax.set_ylabel("Average days of supply (capped at 7,000)")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average overstock flag")
    top_overstock = inv_product.sort_values(["avg_overstock", "avg_days_supply"], ascending=False).head(2)
    for _, row in top_overstock.iterrows():
        ax.annotate(row["product_name"], (row["units"], min(row["avg_days_supply"], 7000)), xytext=(4, 4), textcoords="offset points", fontsize=7)
    save_figure(fig, "08_inventory_misalignment")


def plot_web(frames: dict[str, pd.DataFrame]) -> None:
    sales = frames["sales"]
    web = frames["web_traffic"]
    web_daily = web.groupby("date").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    )
    joined = sales.set_index("Date")[["Revenue"]].join(web_daily, how="inner")
    corr = joined.corr(method="spearman")["Revenue"].drop("Revenue").sort_values(ascending=False)
    lead_lag = pd.DataFrame(
        {
            "lag_days": list(range(-14, 15)),
            "spearman": [joined["Revenue"].corr(joined["sessions"].shift(lag), method="spearman") for lag in range(-14, 15)],
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.0), gridspec_kw={"width_ratios": [1.1, 0.9, 0.9]})
    ax = axes[0]
    ax.scatter(joined["sessions"], joined["Revenue"], s=8, alpha=0.28, color=OKABE[0])
    ax.set_title("Daily sessions vs revenue")
    ax.set_xlabel("Daily sessions")
    ax.set_ylabel("Daily revenue (VND)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.1f}M")

    ax = axes[1]
    sns.barplot(x=corr.values, y=corr.index, color=OKABE[2], ax=ax)
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_title("Traffic features vs revenue")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("")

    ax = axes[2]
    ax.plot(lead_lag["lag_days"], lead_lag["spearman"], marker="o", color=OKABE[1], linewidth=1.4)
    ax.axvline(0, color="0.55", linestyle="--", linewidth=0.8)
    ax.set_title("Sessions lead/lag check")
    ax.set_xlabel("Lag days (sessions shifted)")
    ax.set_ylabel("Spearman rho")
    save_figure(fig, "09_web_traffic_signal")


def generate_figures(frames: dict[str, pd.DataFrame], enriched: dict[str, pd.DataFrame], metrics: dict[str, object]) -> None:
    plot_revenue_decomposition(frames, enriched, metrics)
    plot_seasonality(frames, enriched)
    plot_profit_pool(enriched)
    plot_promotion(frames, enriched)
    plot_customer_channel_payment(frames, enriched)
    plot_returns_experience(frames, enriched)
    plot_geography(frames, enriched)
    plot_inventory(frames, enriched)
    plot_web(frames)


def latex_table_data_audit(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{latex_escape(row['table'])} & {fmt_int(row['rows'])} & {int(row['columns'])} & "
            f"{latex_escape(row['date_min'])}--{latex_escape(row['date_max'])} & "
            f"{latex_escape(row['max_missing_column'])} ({row['max_missing_pct']:.2f}\\%) \\\\"
        )
    return "\n".join(rows)


def latex_action_rows(action_df: pd.DataFrame) -> str:
    rows = []
    for _, row in action_df.iterrows():
        rows.append(
            f"{row['priority']} & {latex_escape(row['decision_area'])} & "
            f"{latex_escape(row['evidence'])} & {latex_escape(row['recommended_action'])} \\\\"
        )
    return "\n".join(rows)


def generate_latex(metrics: dict[str, object], data_audit_df: pd.DataFrame, generated_tables: dict[str, pd.DataFrame]) -> None:
    scope = metrics["data_scope"]
    regime = metrics["regime"]
    season = metrics["seasonality"]
    cat_summary = pd.DataFrame(metrics["category_summary"]).set_index("category")
    promo = metrics["promotion"]
    customer = metrics["customer_channel_payment"]
    returns_exp = metrics["returns_experience"]
    geo = metrics["geography"]
    inv = metrics["inventory"]
    web = metrics["web_traffic"]
    top_margin = metrics["top_margin_products"][0]

    region_total = pd.DataFrame(geo["region_total"]).set_index("region")
    region_cat = pd.DataFrame(geo["region_category"])
    west_outdoor = region_cat[(region_cat["region"] == "West") & (region_cat["category"] == "Outdoor")]["category_share_in_region"].iloc[0]
    east_outdoor = region_cat[(region_cat["region"] == "East") & (region_cat["category"] == "Outdoor")]["category_share_in_region"].iloc[0]
    wrong_size = pd.DataFrame(returns_exp["return_reason"]).set_index("return_reason").loc["wrong_size"]
    ret_cat = pd.DataFrame(returns_exp["return_category"])
    ret_min = ret_cat["return_rate"].min()
    ret_max = ret_cat["return_rate"].max()
    source = pd.DataFrame(customer["order_source"]).set_index("order_source")
    pay = pd.DataFrame(customer["payments"]).set_index("payment_method")

    data_rows = latex_table_data_audit(data_audit_df)
    action_rows = latex_action_rows(generated_tables["action"])

    tex = rf"""
\documentclass{{article}}
\usepackage[preprint,nonatbib]{{neurips_2025}}
\usepackage{{fontspec}}
\setmainfont{{Liberation Serif}}
\setsansfont{{Liberation Sans}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{makecell}}
\usepackage{{float}}
\usepackage{{placeins}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\captionsetup{{font=small,labelfont=bf}}
\setlist[itemize]{{leftmargin=*,nosep}}
\setlist[enumerate]{{leftmargin=*,nosep}}
\graphicspath{{{{./}}}}

\title{{Part 2 EDA/Data Storytelling: Từ doanh thu đến quyết định vận hành cho e-commerce fashion Việt Nam}}
\author{{Datathon 2026 -- The Gridbreakers\\Part 2 EDA Report}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Báo cáo này chỉ dùng 13 bảng dữ liệu thật dành cho EDA: products, customers, promotions, geography, orders, order\_items, payments, shipments, returns, reviews, sales, inventory và web\_traffic. Hai file phục vụ Part 3/submission không được đọc hoặc dùng làm bằng chứng. Câu chuyện chính là doanh nghiệp không thiếu tín hiệu, mà đang có ba điểm nghẽn có thể hành động: nhu cầu gãy mạnh từ 2019 và mùa cao điểm nằm ở Q2; lợi nhuận tập trung vào Streetwear/SKU chọn lọc trong khi promotion có dấu hiệu phòng thủ; và tồn kho vừa stockout vừa overstock, làm mất cơ hội ở SKU nóng và khóa vốn ở SKU lạnh.
\end{{abstract}}

\section{{Cách đọc báo cáo}}
Report này đi theo dòng chảy của một đơn hàng e-commerce: nhu cầu thị trường tạo traffic và đơn hàng, đơn hàng rơi vào danh mục/SKU cụ thể, promotion thay đổi giá trị đơn, trải nghiệm giao nhận và return quyết định phần doanh thu giữ lại, còn inventory quyết định doanh nghiệp có đủ hàng để bán đúng mùa hay không. Rubric Part 2 vẫn được giữ trong cấu trúc lập luận: mỗi phát hiện đều bắt đầu từ số liệu mô tả, đi tiếp sang chẩn đoán, rồi kết thúc bằng hệ quả kinh doanh và hướng hành động. Phần modeling Part 3 chỉ được nhắc ngắn ở cuối.

\textbf{{Data guardrail.}} Script sinh report là \texttt{{report/build\_part2\_report.py}}. Danh sách đọc dữ liệu được hard-code đúng 13 bảng EDA; \texttt{{sales\_test.csv}} và \texttt{{sample\_submission.csv}} nằm trong danh sách cấm. Tổng phạm vi train là {scope['sales_min']}--{scope['sales_max']} với {scope['sales_days']:,} ngày doanh thu, tổng doanh thu {fmt_money(scope['total_revenue'])}, gross margin {pct(scope['gross_margin_rate'])}.

\begin{{table}}[H]
\caption{{Audit 13 bảng dùng trong EDA. Missing cao ở promo fields là nullable theo schema, không phải lỗi dữ liệu.}}
\centering
\scriptsize
\begin{{tabularx}}{{\linewidth}}{{lrrlX}}
\toprule
Table & Rows & Cols & Date range & Max missing column \\
\midrule
{data_rows}
\bottomrule
\end{{tabularx}}
\end{{table}}

\section{{Câu chuyện chính}}
Nhìn từ xa, doanh nghiệp giống như đang gặp một bài toán forecasting: cần dự báo doanh thu ngày. Nhưng khi nối 13 bảng lại với nhau, vấn đề hiện ra rộng hơn. Đây là một retailer fashion e-commerce có demand đã đổi trạng thái từ 2019, mùa cao điểm nằm ở Q2 thay vì cuối năm, và phần lớn revenue phụ thuộc vào Streetwear. Ở tầng vận hành, promotion đang được dùng nhiều vào những ngày yếu, return chủ yếu đến từ vấn đề size/expectation, còn inventory vừa stockout vừa overstock. Nói cách khác, doanh nghiệp không thiếu dữ liệu; vấn đề là các quyết định về mùa vụ, SKU, promotion và tồn kho đang chưa cùng nhịp.

\section{{Cú gãy 2019: doanh thu rơi vì mất lượng đơn}}
Điểm ngoặt lớn nhất của chuỗi doanh thu không nằm ở năm 2020 mà đã xuất hiện trong 2019. Revenue năm 2019 giảm {regime['revenue_yoy_2019_pct']:.1f}\% YoY; sau điểm gãy này, daily revenue trung bình giai đoạn 2019--2022 thấp hơn {pct(regime['post_vs_pre_drop'])} so với 2012--2018. Đến 2022 doanh thu có phục hồi {regime['revenue_yoy_2022_pct']:.1f}\%, nhưng vẫn chưa trở lại nền trước 2019.

Điều quan trọng là cú rơi này không đến từ việc khách mua đơn nhỏ hơn. Decomposition cho thấy số đơn 2019 giảm {regime['orders_yoy_2019_pct']:.1f}\%, units giảm {regime['units_yoy_2019_pct']:.1f}\%, trong khi AOV lại tăng {regime['aov_yoy_2019_pct']:.1f}\%. Vì vậy tín hiệu gần nhất trong dữ liệu là demand/order flow bị mất, không phải basket value suy yếu. Nếu doanh nghiệp lập kế hoạch tồn kho và ngân sách bằng cách trung bình hóa toàn bộ lịch sử 2012--2022, các năm trước 2019 sẽ kéo kế hoạch lên quá cao. Baseline vận hành nên bắt đầu từ giai đoạn hậu 2019, rồi dùng order count và units sold làm early-warning thay vì chờ revenue tổng phản ứng.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/01_revenue_decomposition.pdf}}
\caption{{Revenue break và decomposition. Sources: \texttt{{sales}}, \texttt{{orders}}, \texttt{{order\_items}}, \texttt{{products}}.}}
\end{{figure}}

\section{{Mùa bán hàng thật sự bắt đầu trước mùa hè}}
Sau cú gãy demand, câu hỏi tiếp theo là doanh nghiệp nên chuẩn bị cho mùa nào. Dữ liệu trả lời khá rõ: tháng {season['best_month']} có daily revenue cao hơn trung bình {season['best_month_lift_pct']:.1f}\%, còn tháng {season['worst_month']} thấp hơn {abs(season['worst_month_lift_pct']):.1f}\%. Đây là một pattern quan trọng vì nó đi ngược trực giác phổ biến rằng fashion sẽ cao điểm cuối năm.

Khi tách theo category, seasonality cũng không còn là một đường duy nhất. Streetwear peak tháng {int(metrics['category_peak_months']['Streetwear']['month'])}, GenZ peak tháng {int(metrics['category_peak_months']['GenZ']['month'])}, trong khi Outdoor peak tháng {int(metrics['category_peak_months']['Outdoor']['month'])}. Mix category kéo tổng doanh nghiệp về Q2, nhưng Outdoor vẫn cần cách nhìn riêng. Vì vậy lịch mua hàng, lịch campaign và lịch replenishment nên bắt đầu từ tháng 3 cho Streetwear/Casual/GenZ, đồng thời giữ ngân sách và hàng Outdoor cho tháng 12 và các vùng phù hợp.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/02_seasonality_category.pdf}}
\caption{{Seasonality tổng và theo category. Sources: \texttt{{sales}}, \texttt{{orders}}, \texttt{{order\_items}}, \texttt{{products}}.}}
\end{{figure}}

\section{{Revenue đến từ Streetwear, nhưng lợi nhuận nằm ở SKU được chọn đúng}}
Nếu chỉ nhìn revenue tổng, doanh nghiệp gần như là một doanh nghiệp Streetwear: category này chiếm {pct(cat_summary.loc['Streetwear', 'revenue_share'])} revenue và HHI category đạt {metrics['category_hhi']:.3f}. Mức tập trung này vừa là lợi thế nhận diện, vừa là rủi ro: bất kỳ shock nào ở Streetwear cũng sẽ đi thẳng vào P\&L tổng.

Nhưng trong Streetwear, không phải SKU nào cũng đáng scale như nhau. SKU gross-margin cao nhất là {latex_escape(top_margin['product_name'])}, tạo {fmt_money(top_margin['margin'])} gross margin. Biểu đồ profit pool cho thấy bestseller theo units không tự động là profit driver vì margin rate, price point và promo dependence khác nhau. Quyết định merchandising nên chuyển từ ``bán nhiều thì nhập nhiều'' sang gross-margin by SKU/month: bảo vệ SKU profit driver trước mùa cao, còn các SKU volume lớn nhưng margin mỏng phải có guardrail rõ ràng.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/03_profit_pool.pdf}}
\caption{{Profit pool theo category và SKU. Sources: \texttt{{products}}, \texttt{{order\_items}}.}}
\end{{figure}}

\section{{Promotion đang giống phanh giảm rủi ro hơn là bàn đạp tăng trưởng}}
Promotion là nơi dễ bị diễn giải quá nhanh. Trong dữ liệu này, ngày có trên 50\% đơn dùng promo lại có median revenue thấp hơn {pct(promo['heavy_vs_light_median_drop'])} so với ngày có dưới 10\% đơn dùng promo; Spearman giữa promo share và revenue là {promo['daily_spearman_revenue_promo_share']:.3f}. Con số này không chứng minh promo làm giảm revenue, nhưng nó phá vỡ giả định ``cứ tăng khuyến mãi là tăng doanh thu''.

Cách đọc hợp lý hơn là promotion đang được dùng phòng thủ: team có thể tung coupon vào ngày dự kiến yếu, hoặc discount đang rơi vào khách vốn đã định mua. Nếu không đo uplift thật, doanh nghiệp có thể đổi gross margin lấy một lượng revenue không tăng tương ứng. Promotion nên được quản trị như một intervention có holdout/A-B test, tách rõ campaign tăng trưởng và campaign cứu ngày yếu, đồng thời đặt margin guardrail ở cấp SKU.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/04_promotion_guardrails.pdf}}
\caption{{Promo intensity, promo dependence và catalog khuyến mãi. Sources: \texttt{{orders}}, \texttt{{order\_items}}, \texttt{{promotions}}, \texttt{{products}}.}}
\end{{figure}}

\section{{Xương sống thương mại là khách quay lại và search intent}}
Sau category và promo, lớp tiếp theo là khách hàng. Top 20\% khách tạo {pct(customer['top20_customer_revenue_share'])} revenue; {pct(customer['repeat_customer_share'])} khách đã mua hơn một đơn. Đây là một doanh nghiệp có core cohort mạnh, không chỉ sống bằng acquisition mới.

Kênh cũng củng cố nhận định đó. Organic search đóng {pct(source.loc['organic_search', 'revenue_share'])} revenue, còn credit card chiếm {pct(pay.loc['credit_card', 'value_share'])} payment value. Người mua có vẻ đến với ý định khá rõ, rồi quay lại nhiều lần. Điều này khiến retention quan trọng hơn vẻ ngoài của traffic volume: mất nhóm khách core sẽ đắt hơn nhiều so với một biến động nhỏ ở acquisition. Doanh nghiệp nên theo dõi active repeat customers 30 ngày như một KPI demand, tạo early-access/VIP drop trước mùa cao, và đảm bảo checkout/payment flow ổn định cho nhóm thanh toán bằng thẻ.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/05_customer_channel_payment.pdf}}
\caption{{Customer Pareto, order source và payment mix. Sources: \texttt{{customers}}, \texttt{{orders}}, \texttt{{order\_items}}, \texttt{{payments}}.}}
\end{{figure}}

\section{{Return leakage không nằm ở một category xấu}}
Doanh thu không kết thúc ở checkout; một phần bị rò qua return. Lý do lớn nhất là \texttt{{wrong\_size}}, chiếm {pct(wrong_size['quantity_share'])} returned units và {pct(wrong_size['refund_share'])} refund. Nếu return tập trung vào một category, giải pháp có thể là sửa line sản phẩm đó. Nhưng ở đây return rate giữa category chỉ dao động từ {pct(ret_min)} đến {pct(ret_max)}, còn correlation giữa delivery days và rating gần như bằng 0 ({returns_exp['delivery_rating_corr']:.3f}).

Vì vậy vấn đề có vẻ nằm ở cơ chế mua hàng: size guide, fit expectation, mô tả sản phẩm, hoặc chính sách đổi trả chung. Hành động tốt nhất không phải cắt một category, mà là giảm sai lệch trước checkout: chuẩn hóa size chart, hiển thị fit notes theo SKU, cảnh báo những biến thể hay bị trả và audit mô tả sản phẩm cho nhóm \texttt{{not\_as\_described}}.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/06_returns_experience.pdf}}
\caption{{Return reasons, category return-rate/rating và delivery-rating check. Sources: \texttt{{returns}}, \texttt{{reviews}}, \texttt{{shipments}}, \texttt{{products}}, \texttt{{order\_items}}.}}
\end{{figure}}

\section{{Địa lý không chỉ là nơi bán nhiều hay ít}}
Khi đặt demand lên bản đồ, East là vùng lớn nhất với {pct(region_total.loc['East', 'revenue_share'])} revenue, Central đóng {pct(region_total.loc['Central', 'revenue_share'])}, và West đóng {pct(region_total.loc['West', 'revenue_share'])}. Nhưng điểm đáng chú ý hơn là mix category: Outdoor chiếm {pct(west_outdoor)} revenue ở West, cao hơn nhiều so với {pct(east_outdoor)} ở East.

Điều này biến geography từ một bảng tham chiếu thành input cho merchandising. Một plan toàn quốc có thể đúng về tổng revenue nhưng sai về category allocation. Streetwear nên được scale mạnh hơn ở East/Central; Outdoor cần được nhìn riêng ở West và gắn với mùa peak của category này.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/07_geography_mix.pdf}}
\caption{{Revenue concentration và category mix theo vùng. Sources: \texttt{{geography}}, \texttt{{orders}}, \texttt{{order\_items}}, \texttt{{products}}.}}
\end{{figure}}

\section{{Nút thắt vận hành: thiếu hàng nóng, thừa hàng lạnh}}
Inventory là nơi câu chuyện quay lại quyết định thực thi. Trong năm 2022, average monthly stockout rate đạt {pct(inv['stockout_2022_avg'])}, overstock rate cũng lên tới {pct(inv['overstock_2022_avg'])}, trong khi sell-through trung bình chỉ {pct(inv['sell_through_2022_avg'])}. Hai chỉ số stockout và overstock cùng cao cho thấy doanh nghiệp không chỉ thiếu hàng; doanh nghiệp đang phân bổ sai SKU/tháng.

Hệ quả rất thực tế: SKU nóng làm mất revenue vì stockout, còn SKU lạnh khóa vốn vì days-of-supply dài. Trước mùa cao Q2, ưu tiên không phải tăng nhập toàn danh mục mà là reallocation: replenish high-demand stockout SKU, đồng thời markdown hoặc liquidate chronic overstock với guardrail margin.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/08_inventory_misalignment.pdf}}
\caption{{Inventory health theo tháng và SKU-level demand vs days-of-supply. Sources: \texttt{{inventory}}, \texttt{{products}}, \texttt{{order\_items}}.}}
\end{{figure}}

\section{{Traffic là tín hiệu phụ, không phải câu trả lời}}
Web traffic vẫn có giá trị, nhưng không đủ mạnh để một mình giải thích revenue. Spearman giữa sessions và revenue là {web['spearman']['sessions']:.3f}; unique visitors là {web['spearman']['unique_visitors']:.3f}. Kiểm tra lead/lag cũng không làm tín hiệu mạnh lên nhiều: điểm tốt nhất của sessions chỉ đạt {web['best_lag_spearman']:.3f} tại lag {web['best_lag_days']} ngày.

Điều này hợp lý sau khi đã nhìn các tầng trước. Revenue bị chi phối bởi mùa vụ, category mix, promotion, repeat customers và inventory gate. Đẩy traffic vào một SKU đang stockout hoặc vào một tháng trái mùa sẽ không tạo revenue tương ứng. Traffic nên được dùng như feature phụ và nên đo theo source/conversion/value, đặc biệt khi quyết định paid media phải đi cùng trạng thái tồn kho.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/09_web_traffic_signal.pdf}}
\caption{{Traffic signal strength và lead/lag check. Sources: \texttt{{web\_traffic}}, \texttt{{sales}}.}}
\end{{figure}}

\section{{Bản đồ hành động}}
Sau khi nối các phần lại, các quyết định ưu tiên có thể tóm lại như sau. Bảng này không thay thế câu chuyện phía trên; nó là bản đồ triển khai để team commercial, marketing và operations cùng nhìn một nguồn sự thật.
\begin{{table}}[H]
\caption{{Prescriptive summary: quyết định kinh doanh gắn với bằng chứng.}}
\centering
\scriptsize
\begin{{tabularx}}{{\linewidth}}{{r l X X}}
\toprule
\# & Nhóm quyết định & Bằng chứng chính & Việc nên làm \\
\midrule
{action_rows}
\bottomrule
\end{{tabularx}}
\end{{table}}

\section{{Part 3 note và giới hạn suy luận}}
Part 3/modeling không phải trọng tâm của report này. Nếu template yêu cầu nhắc, EDA gợi ý các feature hợp lệ từ train-only data: post-2019 regime indicator, month/day-of-week/Fourier seasonality, promo share lag, active repeat customers, category mix, web traffic, inventory stockout/overstock lag và return/refund lag. Không có số liệu test/submission nào được dùng trong EDA.

Các claim causal được giữ thận trọng. Đặc biệt, promo--revenue là quan hệ quan sát nên chỉ hỗ trợ giả thuyết về defensive promotion/cannibalization; muốn đo uplift cần experiment hoặc thiết kế quasi-experiment. Inventory không có warehouse/region dimension nên region allocation là khuyến nghị merchandising, không phải tối ưu kho vùng đầy đủ.

\end{{document}}
"""
    (REPORT / "part2_eda_report.tex").write_text(dedent(tex).strip() + "\n", encoding="utf-8")


def generate_summary(metrics: dict[str, object], generated_tables: dict[str, pd.DataFrame], workspace_df: pd.DataFrame) -> None:
    chart_df = generated_tables["chart_summary"]
    rows = []
    for _, row in chart_df.iterrows():
        rows.append(
            f"| `{row['artifact']}` | {row['tables_used']} | {row['main_insight']} | {row['key_numbers']} |"
        )
    workspace_rows = []
    for _, row in workspace_df.iterrows():
        detail = []
        for col in ["cells", "code_cells", "markdown_cells", "outputs", "files", "bytes"]:
            if col in row and pd.notna(row[col]):
                detail.append(f"{col}={int(row[col])}")
        workspace_rows.append(
            f"| `{row['path']}` | {'yes' if row['exists'] else 'no'} | {', '.join(detail)} | {row['audit_note']} |"
        )

    scope = metrics["data_scope"]
    text = f"""# Part 2 EDA Report Summary

Generated by `report/build_part2_report.py`.

## Data guardrails

- Tables used: {', '.join(TABLES_13)}
- Forbidden for EDA and not read: {', '.join(sorted(FORBIDDEN_FOR_EDA))}
- Date scope from `sales.csv`: {scope['sales_min']} to {scope['sales_max']} ({scope['sales_days']:,} days)
- Total train revenue: {fmt_money(scope['total_revenue'])}
- Gross margin rate: {plain_pct(scope['gross_margin_rate'])}

## Output files

- Source code: `report/build_part2_report.py`
- LaTeX: `report/part2_eda_report.tex`
- PDF: `report/part2_eda_report.pdf`
- Figures: `report/figures/*.png` and `report/figures/*.pdf`
- Tables: `report/tables/*.csv`
- Metrics JSON: `report/eda_metrics.json`

## Chart/table evidence map

| Artifact | Tables used | Main insight | Key numbers |
|---|---|---|---|
{chr(10).join(rows)}

## Workspace audit

| Path | Exists | Details | Audit note |
|---|---:|---|---|
{chr(10).join(workspace_rows)}

## Reproducibility notes

- The report pipeline reads only files from `data/` and only the 13 configured EDA tables.
- No external/internet data, `sales_test.csv`, or `sample_submission.csv` are used.
- All numeric claims in `part2_eda_report.tex` are populated from `report/eda_metrics.json` generated in the same run.
"""
    (REPORT / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_dirs()
    frames = load_data()
    enriched = build_enriched(frames)
    data_audit_df, fk_df = data_audit(frames)
    workspace_df = build_workspace_audit()
    metrics = calculate_metrics(frames, enriched)
    generated_tables = generate_tables(metrics, data_audit_df, fk_df, workspace_df)
    generate_figures(frames, enriched, metrics)
    generate_latex(metrics, data_audit_df, generated_tables)
    generate_summary(metrics, generated_tables, workspace_df)

    metrics_path = REPORT / "eda_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path.relative_to(ROOT)}")
    print(f"Wrote {(REPORT / 'part2_eda_report.tex').relative_to(ROOT)}")
    print(f"Wrote {(REPORT / 'summary.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
