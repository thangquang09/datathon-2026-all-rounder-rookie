"""Additional analyses for report.md.

Produces targeted visualizations in images/report/ and prints
numeric summaries that back up each insight.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")

DATA = Path("data")
OUT = Path("images/report")
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def save(name: str) -> None:
    p = OUT / name
    plt.savefig(p)
    plt.close()
    print(f"saved -> {p}")


print("Loading core tables...")
sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
returns = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
reviews = pd.read_csv(DATA / "reviews.csv", parse_dates=["review_date"])
shipments = pd.read_csv(DATA / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])

# =======================================================================
# 1. Revenue trend + 30/90-day rolling + COVID annotation
# =======================================================================
print("\n== Insight 1: Revenue trend, volatility, regime changes ==")
sales["rev_30"] = sales["Revenue"].rolling(30, min_periods=1).mean()
sales["rev_90"] = sales["Revenue"].rolling(90, min_periods=1).mean()
sales["cv_30"] = (
    sales["Revenue"].rolling(30, min_periods=7).std()
    / sales["Revenue"].rolling(30, min_periods=7).mean()
)

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(sales["Date"], sales["Revenue"], color="lightsteelblue", lw=0.5, label="Daily Revenue")
ax.plot(sales["Date"], sales["rev_30"], color="tab:blue", lw=1.2, label="30-day MA")
ax.plot(sales["Date"], sales["rev_90"], color="tab:red", lw=1.5, label="90-day MA")
for y, label, color in [
    ("2020-01-23", "COVID-19 outbreak", "grey"),
    ("2021-07-09", "VN Delta wave", "purple"),
]:
    d = pd.Timestamp(y)
    ax.axvline(d, ls="--", color=color, alpha=0.6)
    ax.text(d, sales["Revenue"].max() * 0.98, f" {label}", color=color, fontsize=9, rotation=90, va="top")
ax.set_title("Daily Revenue with 30/90-day moving averages (2012–2022)")
ax.set_ylabel("Revenue (VND)")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save("revenue_trend_with_events.png")

# Annual summary
annual = sales.assign(year=sales["Date"].dt.year).groupby("year").agg(
    Revenue=("Revenue", "sum"), COGS=("COGS", "sum")
)
annual["margin_pct"] = (annual["Revenue"] - annual["COGS"]) / annual["Revenue"] * 100
annual["yoy_%"] = annual["Revenue"].pct_change() * 100
print(annual.round(2))

# =======================================================================
# 2. Seasonality fingerprint — Month × Day-of-week
# =======================================================================
print("\n== Insight 2: Seasonality fingerprint ==")
sales["month"] = sales["Date"].dt.month
sales["dow"] = sales["Date"].dt.dayofweek  # 0=Mon
pivot = sales.pivot_table(index="dow", columns="month", values="Revenue", aggfunc="mean")
# Normalize by row-col global mean to show % deviation
overall = sales["Revenue"].mean()
pivot_pct = (pivot / overall - 1) * 100

fig, ax = plt.subplots(figsize=(12, 4.5))
sns.heatmap(
    pivot_pct,
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".0f",
    cbar_kws={"label": "% deviation from overall daily mean"},
    ax=ax,
    yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
)
ax.set_title("Seasonality fingerprint: mean Revenue deviation (%) by day-of-week × month")
ax.set_xlabel("Month")
ax.set_ylabel("")
save("seasonality_dow_month_heatmap.png")
print("Top (dow,month) cells vs mean (%):")
print(pivot_pct.stack().sort_values(ascending=False).head(8).round(1))
print("Bottom (dow,month) cells vs mean (%):")
print(pivot_pct.stack().sort_values().head(5).round(1))

# =======================================================================
# 3. Promotion effectiveness
# =======================================================================
print("\n== Insight 3: Promotion effectiveness on daily revenue ==")
# Flag days where at least one order had a promo
items["has_promo"] = items["promo_id"].notna()
order_promo = items.groupby("order_id")["has_promo"].any().rename("order_has_promo")
orders2 = orders.merge(order_promo, on="order_id", how="left").fillna({"order_has_promo": False})
daily_promo_share = (
    orders2.groupby(orders2["order_date"].dt.normalize())["order_has_promo"].mean()
    .rename("promo_order_share")
)
daily = sales.set_index("Date")[["Revenue"]].join(daily_promo_share).dropna()
daily["bucket"] = pd.cut(
    daily["promo_order_share"],
    bins=[-0.01, 0.1, 0.25, 0.5, 1.01],
    labels=["<10%", "10–25%", "25–50%", ">50%"],
)
bucket_stats = daily.groupby("bucket")["Revenue"].agg(["count", "mean", "median"]).round(0)
print(bucket_stats)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=daily, x="bucket", y="Revenue", ax=ax, palette="coolwarm", showfliers=False)
ax.set_title("Daily Revenue by share of orders using a promotion")
ax.set_xlabel("Share of orders on that day that used a promo")
ax.set_ylabel("Daily Revenue (VND)")
save("promo_share_vs_daily_revenue.png")

# =======================================================================
# 4. Customer cohort / repeat-buyer concentration
# =======================================================================
print("\n== Insight 4: Customer concentration (Pareto) ==")
# Revenue per order
items["line_revenue"] = items["quantity"] * items["unit_price"]
order_rev = items.groupby("order_id")["line_revenue"].sum().rename("order_revenue")
orders3 = orders.merge(order_rev, on="order_id", how="left").fillna({"order_revenue": 0})
customer_rev = orders3.groupby("customer_id")["order_revenue"].sum().sort_values(ascending=False)
n_customers = len(customer_rev)
cum = customer_rev.cumsum() / customer_rev.sum()
# Top 10 / 20 / 40 %
for q in [0.1, 0.2, 0.4]:
    n = max(int(n_customers * q), 1)
    print(f"Top {int(q*100):>2}% customers ({n:,}) → {cum.iloc[n-1]*100:.1f}% of revenue")

order_count = orders.groupby("customer_id").size()
repeat_share = (order_count > 1).mean() * 100
print(f"Customers with >1 order: {repeat_share:.1f}%")
print(f"Mean / median orders per customer: {order_count.mean():.2f} / {order_count.median():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].plot(np.arange(1, n_customers + 1) / n_customers * 100, cum.values * 100, color="tab:blue")
axes[0].axhline(80, ls="--", color="grey"); axes[0].axvline(20, ls="--", color="grey")
axes[0].set_xlabel("% of customers (sorted by revenue desc)")
axes[0].set_ylabel("% of cumulative revenue")
axes[0].set_title("Pareto curve: customer revenue concentration")

sns.histplot(order_count.clip(upper=10), bins=np.arange(0.5, 11.5, 1), ax=axes[1], color="steelblue")
axes[1].set_xlabel("Orders per customer (clipped at 10)")
axes[1].set_title("Distribution of orders per customer")
save("customer_pareto_and_repeat.png")

# =======================================================================
# 5. Return rate by category + reason financial impact
# =======================================================================
print("\n== Insight 5: Return rate & refund impact by category ==")
items_cat = items.merge(products[["product_id", "category"]], on="product_id", how="left")
sold_qty = items_cat.groupby("category")["quantity"].sum()
ret_cat = (
    returns.merge(products[["product_id", "category"]], on="product_id", how="left")
    .groupby("category")
    .agg(return_qty=("return_quantity", "sum"), refund=("refund_amount", "sum"))
)
impact = sold_qty.to_frame("sold_qty").join(ret_cat)
impact["return_rate_%"] = impact["return_qty"] / impact["sold_qty"] * 100
impact = impact.sort_values("return_rate_%", ascending=False)
print(impact.round(2))

fig, ax = plt.subplots(figsize=(11, 5))
order_cat = impact.index.tolist()
ax.bar(order_cat, impact["return_rate_%"], color=sns.color_palette("rocket", len(order_cat)))
ax.set_ylabel("Return rate (%)")
ax.set_title("Return rate by product category (return_qty / sold_qty)")
ax.tick_params(axis="x", rotation=25)
for i, v in enumerate(impact["return_rate_%"]):
    ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
save("return_rate_by_category.png")

# =======================================================================
# 6. Web traffic → Revenue correlation
# =======================================================================
print("\n== Insight 6: Web traffic vs Revenue correlation ==")
web_daily = web.groupby(web["date"].dt.normalize()).agg(
    sessions=("sessions", "sum"),
    unique_visitors=("unique_visitors", "sum"),
    page_views=("page_views", "sum"),
    bounce_rate=("bounce_rate", "mean"),
    avg_dur=("avg_session_duration_sec", "mean"),
)
joined = sales.set_index("Date")[["Revenue"]].join(web_daily, how="inner")
print(f"Joined rows: {len(joined):,}")
corr = joined.corr(method="spearman")["Revenue"].drop("Revenue").sort_values(ascending=False)
print("Spearman correlation with Revenue:")
print(corr.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(joined["sessions"], joined["Revenue"], s=5, alpha=0.3, color="tab:blue")
axes[0].set_xlabel("Daily sessions")
axes[0].set_ylabel("Daily Revenue")
axes[0].set_title(f"Revenue vs Sessions  (ρ={corr['sessions']:.2f})")

sns.barplot(x=corr.index, y=corr.values, ax=axes[1], palette="vlag")
axes[1].set_ylabel("Spearman ρ with Revenue")
axes[1].set_title("Web-traffic signals vs daily Revenue")
axes[1].tick_params(axis="x", rotation=30)
save("webtraffic_vs_revenue.png")

# =======================================================================
# 7. Inventory health: stockout ↔ lost potential
# =======================================================================
print("\n== Insight 7: Stockout prevalence ==")
inv_summary = (
    inventory.groupby(["year", "month"])
    .agg(
        stockout_rate=("stockout_flag", "mean"),
        overstock_rate=("overstock_flag", "mean"),
        mean_fill_rate=("fill_rate", "mean"),
    )
    .reset_index()
)
inv_summary["date"] = pd.to_datetime(dict(year=inv_summary["year"], month=inv_summary["month"], day=1))
print(inv_summary.tail(12).round(3))

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(inv_summary["date"], inv_summary["stockout_rate"] * 100, color="tab:red", label="Stockout %")
ax.plot(inv_summary["date"], inv_summary["overstock_rate"] * 100, color="tab:orange", label="Overstock %")
ax.plot(inv_summary["date"], inv_summary["mean_fill_rate"] * 100, color="tab:green", label="Mean fill rate %")
ax.set_title("Monthly inventory health (SKUs flagged / mean fill rate)")
ax.set_ylabel("%")
ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save("inventory_health_monthly.png")

# =======================================================================
# 8. Delivery SLA vs rating
# =======================================================================
print("\n== Insight 8: Delivery time vs review rating ==")
shipments["delivery_days"] = (shipments["delivery_date"] - shipments["ship_date"]).dt.days
ship_rev = shipments.merge(reviews[["order_id", "rating"]], on="order_id", how="inner")
ship_rev = ship_rev[ship_rev["delivery_days"].between(0, 30)]
bin_edges = [-0.01, 2, 4, 6, 9, 30]
ship_rev["bucket"] = pd.cut(ship_rev["delivery_days"], bins=bin_edges, labels=["0–2", "3–4", "5–6", "7–9", "10+"])
sla = ship_rev.groupby("bucket").agg(
    n=("rating", "size"), mean_rating=("rating", "mean"), low_rating_share=("rating", lambda s: (s <= 2).mean())
)
print(sla.round(3))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(sla.index.astype(str), sla["mean_rating"], color="steelblue", alpha=0.8, label="Mean rating")
ax1.set_ylabel("Mean rating", color="steelblue")
ax1.set_ylim(1, 5)
ax2 = ax1.twinx()
ax2.plot(sla.index.astype(str), sla["low_rating_share"] * 100, color="tomato", marker="o", label="% rating ≤ 2")
ax2.set_ylabel("% of reviews with rating ≤ 2", color="tomato")
ax1.set_xlabel("Delivery time (days)")
ax1.set_title("Delivery time vs customer rating")
save("delivery_vs_rating.png")

# =======================================================================
# 9. Forecast-ready features: YoY correction + regime break detection
# =======================================================================
print("\n== Insight 9: Train/test regime gap (forecast horizon ~540 days) ==")
# Revenue per quarter to see recent regime
sales["q"] = sales["Date"].dt.to_period("Q").dt.to_timestamp()
q = sales.groupby("q")["Revenue"].sum()
print("Latest 8 quarters:")
print(q.tail(8).round(0))

# Train vs test size
train_days = len(sales)
test_days = (pd.Timestamp("2024-07-01") - pd.Timestamp("2023-01-01")).days + 1
print(f"Train days: {train_days:,} | Test days: {test_days}")

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(q.index, q.values, width=80, color="steelblue", alpha=0.8)
ax.set_title("Quarterly Revenue (train horizon only)")
ax.set_ylabel("Revenue")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
# Highlight test period on x-axis
ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-07-01"), alpha=0.15, color="tomato", label="Test horizon (to forecast)")
ax.legend()
save("quarterly_revenue_with_test_horizon.png")

print("\nDone.")
