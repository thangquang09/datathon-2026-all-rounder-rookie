"""Independent metric audit for the Phase 2 product insight notebook.

This script recomputes core metrics from raw CSV files and compares them with
the notebook-exported tables. It is intentionally separate from the notebook
builder so arithmetic mistakes are easier to catch.

Run from the repository root:
    uv run python group_job/audit_phase2_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "group_job"
TABLE = WORK / "tables"
REPORT = WORK / "phase2_metric_audit.md"


def load_csv(name: str, parse_dates: list[str] | None = None, dtype: dict | None = None) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", parse_dates=parse_dates or [], dtype=dtype, low_memory=False)


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    return float((left.astype(float) - right.astype(float)).abs().max())


def assert_close(name: str, actual: float, expected: float, tolerance: float = 0.01) -> None:
    diff = abs(float(actual) - float(expected))
    if diff > tolerance:
        raise AssertionError(f"{name}: actual={actual}, expected={expected}, diff={diff}, tolerance={tolerance}")


orders = load_csv("orders", ["order_date"])
items = load_csv("order_items", dtype={"promo_id": "string", "promo_id_2": "string"})
products = load_csv("products")
sales = load_csv("sales", ["Date"])
returns = load_csv("returns", ["return_date"])
reviews = load_csv("reviews", ["review_date"])
inventory = load_csv("inventory", ["snapshot_date"])
geography = load_csv("geography")

fact = (
    items.merge(
        orders[["order_id", "order_date", "customer_id", "zip", "order_status", "payment_method", "device_type", "order_source"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    .merge(products, on="product_id", how="left", validate="many_to_one")
)
fact["line_revenue"] = fact["quantity"] * fact["unit_price"]
fact["line_cogs"] = fact["quantity"] * fact["cogs"]
fact["gross_margin"] = fact["line_revenue"] - fact["line_cogs"]
fact["promo_used"] = fact[["promo_id", "promo_id_2"]].notna().any(axis=1)
fact["promo_revenue"] = np.where(fact["promo_used"], fact["line_revenue"], 0.0)
fact["month"] = fact["order_date"].dt.month


checks: list[dict] = []


def record(name: str, value, expected=None, status: str = "PASS", note: str = "") -> None:
    checks.append({"check": name, "value": value, "expected": expected, "status": status, "note": note})


# 1. Join integrity and aggregate consistency.
record("order_items rows after joins", len(fact), len(items))
record("missing order join rows", int(fact["order_date"].isna().sum()), 0)
record("missing product join rows", int(fact["product_name"].isna().sum()), 0)
if len(fact) != len(items):
    raise AssertionError("Fact table row count changed after many-to-one joins.")
if fact["order_date"].isna().any() or fact["product_name"].isna().any():
    raise AssertionError("Fact table has missing order/product joins.")

daily = fact.groupby("order_date", as_index=False).agg(revenue=("line_revenue", "sum"), cogs=("line_cogs", "sum"))
daily_check = sales.merge(daily, left_on="Date", right_on="order_date", how="left")
rev_diff = max_abs_diff(daily_check["Revenue"], daily_check["revenue"])
cogs_diff = max_abs_diff(daily_check["COGS"], daily_check["cogs"])
record("max daily Revenue diff vs sales.csv", rev_diff, 0.0, note="Floating-point rounding only.")
record("max daily COGS diff vs sales.csv", cogs_diff, 0.0, note="Floating-point rounding only.")
if rev_diff > 0.02 or cogs_diff > 0.02:
    raise AssertionError("Fact table does not reconcile with sales.csv.")


# 2. Exported product/category tables match independent recomputation.
export_product = pd.read_csv(TABLE / "03_product_summary.csv")
export_category = pd.read_csv(TABLE / "02_category_summary.csv")

return_product = returns.groupby("product_id", as_index=False).agg(return_qty=("return_quantity", "sum"), refund_amount=("refund_amount", "sum"))
review_product = reviews.groupby("product_id", as_index=False).agg(avg_rating=("rating", "mean"), n_reviews=("rating", "size"))
product_recomputed = (
    fact.groupby(["product_id", "product_name", "category", "segment", "size", "color"], as_index=False)
    .agg(
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        revenue=("line_revenue", "sum"),
        gross_margin=("gross_margin", "sum"),
        promo_revenue=("promo_revenue", "sum"),
    )
    .merge(return_product, on="product_id", how="left")
    .merge(review_product, on="product_id", how="left")
)
product_recomputed[["return_qty", "refund_amount", "promo_revenue"]] = product_recomputed[
    ["return_qty", "refund_amount", "promo_revenue"]
].fillna(0)
product_cmp = export_product.merge(
    product_recomputed,
    on=["product_id", "product_name", "category", "segment", "size", "color"],
    suffixes=("_export", "_audit"),
    validate="one_to_one",
)
record("product rows exported", len(export_product), len(product_recomputed))
if len(product_cmp) != len(product_recomputed):
    raise AssertionError("Product table export does not match recomputed product rows.")
for metric in ["units", "orders", "revenue", "gross_margin", "promo_revenue", "return_qty", "refund_amount"]:
    diff = max_abs_diff(product_cmp[f"{metric}_export"], product_cmp[f"{metric}_audit"])
    record(f"product table max diff: {metric}", diff, 0.0)
    if diff > 0.02:
        raise AssertionError(f"Product metric mismatch: {metric}")

category_base = (
    fact.groupby("category", as_index=False)
    .agg(
        products=("product_id", "nunique"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        revenue=("line_revenue", "sum"),
        gross_margin=("gross_margin", "sum"),
        promo_revenue=("promo_revenue", "sum"),
    )
)
category_returns = (
    returns.merge(products[["product_id", "category"]], on="product_id", how="left")
    .groupby("category", as_index=False)
    .agg(return_qty=("return_quantity", "sum"), refund_amount=("refund_amount", "sum"))
)
category_recomputed = category_base.merge(category_returns, on="category", how="left").fillna(0)
category_cmp = export_category.merge(category_recomputed, on="category", suffixes=("_export", "_audit"), validate="one_to_one")
for metric in ["products", "units", "orders", "revenue", "gross_margin", "promo_revenue", "return_qty", "refund_amount"]:
    diff = max_abs_diff(category_cmp[f"{metric}_export"], category_cmp[f"{metric}_audit"])
    record(f"category table max diff: {metric}", diff, 0.0)
    if diff > 0.02:
        raise AssertionError(f"Category metric mismatch: {metric}")


# 3. Validate headline insight values.
top_category = category_recomputed.sort_values("gross_margin", ascending=False).iloc[0]
top_product = product_recomputed.sort_values("gross_margin", ascending=False).iloc[0]
promo_revenue_share = fact["promo_revenue"].sum() / fact["line_revenue"].sum()
promo_line_share = fact["promo_used"].mean()
return_reason = returns["return_reason"].value_counts(normalize=True)
wrong_size_share = float(return_reason.loc["wrong_size"])

record("top category by gross margin", top_category["category"], "Streetwear")
record("top category gross margin", round(float(top_category["gross_margin"]), 2), 1738676765.00)
record("top product by gross margin", top_product["product_name"], "SaigonFlex UM-43")
record("top product gross margin", round(float(top_product["gross_margin"]), 2), 130456863.28)
record("promo revenue share", round(float(promo_revenue_share), 6), 0.330814)
record("promo line share", round(float(promo_line_share), 6), 0.386635)
record("wrong_size return share", round(wrong_size_share, 6), 0.349708)

if top_category["category"] != "Streetwear":
    raise AssertionError("Top category headline changed.")
if top_product["product_name"] != "SaigonFlex UM-43":
    raise AssertionError("Top product headline changed.")
assert_close("promo revenue share", promo_revenue_share, 0.33081435613479904, 1e-9)
assert_close("promo line share", promo_line_share, 0.3866349316956521, 1e-9)
assert_close("top product gross margin", float(top_product["gross_margin"]), 130456863.28, 0.01)
assert_close("wrong_size return share", wrong_size_share, 0.3497083051653772, 1e-9)


# 4. Validate dimension limits stated in the notebook.
record("inventory product coverage", f"{inventory['product_id'].nunique()} / {products['product_id'].nunique()}", note="Inventory is monthly and does not cover every catalog product.")
record("geography join missing rows", int(fact.merge(geography[["zip", "region"]], on="zip", how="left")["region"].isna().sum()), 0)


audit_df = pd.DataFrame(checks)
audit_df.to_csv(TABLE / "16_metric_audit.csv", index=False)

lines = [
    "# Phase 2 Metric Audit",
    "",
    "Independent recomputation from raw CSV files completed successfully.",
    "",
    "| Check | Value | Expected | Status | Note |",
    "|---|---:|---:|---|---|",
]
for row in checks:
    lines.append(
        f"| {row['check']} | {row['value']} | {'' if row['expected'] is None else row['expected']} | {row['status']} | {row['note']} |"
    )
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({"status": "PASS", "checks": len(checks), "report": str(REPORT)}, ensure_ascii=False, indent=2))
