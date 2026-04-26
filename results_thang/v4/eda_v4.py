"""EDA v4 pipeline — Datathon 2026 "The Gridbreakers".

Idempotent script that:
    1. Loads the 14 CSV tables from ``data/``.
    2. Builds the daily revenue panel with exogenous features.
    3. Runs 4 statistical tests used in the report.
    4. Persists all quoted numbers to ``results/v4/metrics_v4.json``.
    5. Renders 5 publication-quality multi-panel figures (PDF + PNG, 300 DPI).

Run with::

    uv run python results/v4/eda_v4.py

The script is safe to re-run; outputs are overwritten atomically.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & global style
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data"
OUT = HERE
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#000000",  # black
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # TrueType for embedding
    "ps.fonttype": 42,
})


METRICS: dict[str, Any] = {}


def log(key: str, value: Any) -> None:
    """Append a json-safe metric."""
    def _safe(v: Any) -> Any:
        if isinstance(v, (np.floating, np.float64)):
            fv = float(v)
            return fv if np.isfinite(fv) else None
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, np.ndarray):
            return [_safe(x) for x in v.tolist()]
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        if isinstance(v, dict):
            return {str(k): _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        return v

    METRICS[key] = _safe(value)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF (vector, for LaTeX) and PNG (preview)."""
    fig.savefig(IMG / f"{name}.pdf")
    fig.savefig(IMG / f"{name}.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Load tables
# ---------------------------------------------------------------------------

print("[1/6] Loading tables ...", flush=True)

sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
shipments = pd.read_csv(DATA / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
returns = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
reviews = pd.read_csv(DATA / "reviews.csv", parse_dates=["review_date"])
inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
geography = pd.read_csv(DATA / "geography.csv")

log("train_days", int(sales.shape[0]))
log("train_start", sales["Date"].min().isoformat())
log("train_end", sales["Date"].max().isoformat())
log("test_horizon_days", 548)
log("test_range", ["2023-01-01", "2024-07-01"])

# ---------------------------------------------------------------------------
# 2. Build daily panel with exogenous features
# ---------------------------------------------------------------------------

print("[2/6] Building daily panel ...", flush=True)

# Orders per day
orders["order_date"] = pd.to_datetime(orders["order_date"]).dt.normalize()
n_orders = orders.groupby("order_date").size().rename("n_orders")

# Active customers (last 30 days rolling) — computed in a leakage-free way
cust_daily = orders.groupby("order_date")["customer_id"].nunique().rename("unique_customers_day")

# Order-level revenue + category breakdown (reconstruct from order_items for checks)
oi = order_items.merge(products[["product_id", "category", "segment", "price", "cogs"]], on="product_id", how="left")
oi["gross"] = oi["unit_price"] * oi["quantity"]
oi["discount"] = oi["discount_amount"].fillna(0.0)
oi = oi.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
oi["order_date"] = pd.to_datetime(oi["order_date"]).dt.normalize()

# Category × daily revenue (use gross - discount as net item revenue proxy)
oi["net_item"] = oi["gross"] - oi["discount"]
cat_daily = oi.groupby(["order_date", "category"])["net_item"].sum().unstack(fill_value=0.0)

# Promo usage: share of orders that had at least one promo-linked item per day
oi["has_promo"] = oi["promo_id"].notna().astype(int)
promo_order = oi.groupby(["order_date", "order_id"])["has_promo"].max().reset_index()
promo_share = promo_order.groupby("order_date")["has_promo"].mean().rename("promo_share")
total_discount = oi.groupby("order_date")["discount"].sum().rename("total_discount")

# Returns & refund per day
returns["return_date"] = pd.to_datetime(returns["return_date"]).dt.normalize()
refund_daily = returns.groupby("return_date")["refund_amount"].sum().rename("refund")

# Web traffic daily
web["date"] = pd.to_datetime(web["date"]).dt.normalize()
web_daily = web.groupby("date").agg(
    sessions=("sessions", "sum"),
    unique_visitors=("unique_visitors", "sum"),
    page_views=("page_views", "sum"),
    bounce_rate=("bounce_rate", "mean"),
    avg_sess=("avg_session_duration_sec", "mean"),
).reset_index().rename(columns={"date": "order_date"}).set_index("order_date")

# Inventory monthly: stockout & overstock rates
inv_month = inventory.copy()
inv_month["month"] = inv_month["snapshot_date"].dt.to_period("M").dt.to_timestamp()
inv_agg = inv_month.groupby("month").agg(
    stockout_rate=("stockout_flag", "mean"),
    overstock_rate=("overstock_flag", "mean"),
    fill_rate=("fill_rate", "mean"),
    days_of_supply=("days_of_supply", "mean"),
).reset_index()

# Build daily panel
panel = sales.copy().set_index("Date")
panel = panel.join(n_orders, how="left")
panel = panel.join(promo_share, how="left")
panel = panel.join(total_discount, how="left")
panel = panel.join(refund_daily, how="left")
panel = panel.join(web_daily, how="left")
panel = panel.join(cust_daily, how="left")

panel[["n_orders", "total_discount", "refund", "sessions", "unique_visitors", "page_views", "unique_customers_day"]] = (
    panel[["n_orders", "total_discount", "refund", "sessions", "unique_visitors", "page_views", "unique_customers_day"]]
    .fillna(0.0)
)
panel["promo_share"] = panel["promo_share"].fillna(0.0)
panel["bounce_rate"] = panel["bounce_rate"].fillna(panel["bounce_rate"].median())
panel["avg_sess"] = panel["avg_sess"].fillna(panel["avg_sess"].median())
panel["year"] = panel.index.year
panel["month"] = panel.index.month
panel["dow"] = panel.index.dayofweek  # Monday=0
panel["margin"] = (panel["Revenue"] - panel["COGS"]) / panel["Revenue"].replace(0, np.nan)
panel = panel.reset_index().rename(columns={"Date": "date"})

log("panel_rows", int(panel.shape[0]))
log("panel_cols", int(panel.shape[1]))

# ---------------------------------------------------------------------------
# 3. Statistical tests
# ---------------------------------------------------------------------------

print("[3/6] Running statistical tests ...", flush=True)

# --- Test A: Regime break 2019 (Welch t-test pre vs post)
pre = panel[panel["year"] <= 2018]["Revenue"].values
post = panel[panel["year"] >= 2019]["Revenue"].values
t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
pooled_sd = np.sqrt((np.var(pre, ddof=1) + np.var(post, ddof=1)) / 2)
cohen_d = float((np.mean(pre) - np.mean(post)) / pooled_sd)
drop_pct = float((np.mean(pre) - np.mean(post)) / np.mean(pre) * 100)

log("regime.pre_mean_vnd_day", float(np.mean(pre)))
log("regime.post_mean_vnd_day", float(np.mean(post)))
log("regime.drop_pct", drop_pct)
log("regime.t_stat", float(t_stat))
log("regime.p_value", float(p_val))
log("regime.cohen_d", cohen_d)
log("regime.n_pre", int(len(pre)))
log("regime.n_post", int(len(post)))

# Year-on-year annual totals (millions VND)
yoy = panel.groupby("year").agg(
    revenue_total=("Revenue", "sum"),
    days=("Revenue", "size"),
    margin=("margin", "mean"),
).reset_index()
yoy["revenue_mil"] = yoy["revenue_total"] / 1e6
yoy["yoy_pct"] = yoy["revenue_total"].pct_change() * 100
log("yoy_table", yoy[["year", "revenue_mil", "yoy_pct", "margin"]].round(2).to_dict(orient="records"))

# --- Test B: Monthly seasonality
month_mean = panel.groupby("month")["Revenue"].mean()
grand_mean = panel["Revenue"].mean()
month_delta_pct = ((month_mean - grand_mean) / grand_mean * 100).round(2)
log("seasonality.grand_mean_vnd", float(grand_mean))
log("seasonality.month_mean_vnd", month_mean.round(0).to_dict())
log("seasonality.month_delta_pct", month_delta_pct.to_dict())
log("seasonality.top_month", int(month_mean.idxmax()))
log("seasonality.top_month_pct", float(month_delta_pct.max()))
log("seasonality.bottom_month", int(month_mean.idxmin()))
log("seasonality.bottom_month_pct", float(month_delta_pct.min()))

# Month × dow heatmap (percent deviation from grand mean)
mdow = panel.groupby(["dow", "month"])["Revenue"].mean().unstack("month")
mdow_pct = (mdow - grand_mean) / grand_mean * 100
log("seasonality.month_dow_pct_max", float(mdow_pct.values.max()))
log("seasonality.month_dow_pct_min", float(mdow_pct.values.min()))

# --- Test C: Category concentration (HHI)
cat_rev = oi.groupby("category")["net_item"].sum().sort_values(ascending=False)
cat_share = cat_rev / cat_rev.sum()
hhi_cat = float((cat_share ** 2).sum())
log("category.revenue_mil", (cat_rev / 1e6).round(1).to_dict())
log("category.share_pct", (cat_share * 100).round(1).to_dict())
log("category.hhi", hhi_cat)
log("category.top_share_pct", float(cat_share.max() * 100))

# --- Test D: Promo paradox (Mann-Whitney U)
bins = [-0.001, 0.1, 0.25, 0.5, 1.01]
labels = ["<10%", "10-25%", "25-50%", ">50%"]
panel["promo_bucket"] = pd.cut(panel["promo_share"], bins=bins, labels=labels)
bucket_stats = panel.groupby("promo_bucket", observed=True)["Revenue"].agg(["count", "mean", "median"]).reset_index()
log("promo.bucket_table", bucket_stats.round(0).to_dict(orient="records"))

light = panel.loc[panel["promo_bucket"] == "<10%", "Revenue"].values
heavy = panel.loc[panel["promo_bucket"] == ">50%", "Revenue"].values
u_stat, u_p = stats.mannwhitneyu(light, heavy, alternative="two-sided")
light_med = float(np.median(light))
heavy_med = float(np.median(heavy))
gap_pct = float((light_med - heavy_med) / light_med * 100)
log("promo.u_stat", float(u_stat))
log("promo.p_value", float(u_p))
log("promo.light_median", light_med)
log("promo.heavy_median", heavy_med)
log("promo.gap_median_pct", gap_pct)
log("promo.n_light", int(len(light)))
log("promo.n_heavy", int(len(heavy)))

spearman_promo = stats.spearmanr(panel["promo_share"], panel["Revenue"]).statistic
log("promo.spearman_rho", float(spearman_promo))

# --- Test E: Exogenous signal ranking (Spearman with Revenue, leakage-aware)
# Same-day features purely for ranking — these will be lagged before modeling.
exo_cols = [
    "COGS", "n_orders", "refund", "sessions", "unique_visitors", "page_views",
    "bounce_rate", "avg_sess", "promo_share", "total_discount", "margin",
    "unique_customers_day",
]
ranks = {}
for c in exo_cols:
    r = stats.spearmanr(panel[c].fillna(0), panel["Revenue"]).statistic
    ranks[c] = float(r)
# Merge monthly inventory lag-30
inv_agg_lag = inv_agg.copy()
inv_agg_lag["month"] = inv_agg_lag["month"] + pd.offsets.MonthBegin(1)  # shift forward 1 month
inv_agg_lag = inv_agg_lag.rename(columns={
    "stockout_rate": "stockout_rate_lag30",
    "overstock_rate": "overstock_rate_lag30",
    "fill_rate": "fill_rate_lag30",
})[["month", "stockout_rate_lag30", "overstock_rate_lag30", "fill_rate_lag30"]]
panel_m = panel.copy()
panel_m["month_period"] = panel_m["date"].dt.to_period("M").dt.to_timestamp()
panel_m = panel_m.merge(inv_agg_lag, left_on="month_period", right_on="month", how="left")
for c in ["stockout_rate_lag30", "overstock_rate_lag30", "fill_rate_lag30"]:
    sub = panel_m[[c, "Revenue"]].dropna()
    if len(sub) > 30:
        ranks[c] = float(stats.spearmanr(sub[c], sub["Revenue"]).statistic)
log("exogenous.spearman", ranks)

# Lead-lag for web traffic (±7 days)
leads = {}
for k in range(-7, 8):
    shifted = panel["sessions"].shift(-k)  # negative k => traffic leads revenue
    valid = (~shifted.isna()) & (~panel["Revenue"].isna())
    leads[k] = float(stats.spearmanr(shifted[valid], panel.loc[valid, "Revenue"]).statistic)
log("exogenous.lead_lag_sessions", leads)

# --- Test F: Inventory paradox — 2022 monthly table
inv_2022 = inv_agg[inv_agg["month"].dt.year == 2022].copy()
inv_2022["month_str"] = inv_2022["month"].dt.strftime("%Y-%m")
log("inventory.2022_monthly", inv_2022[[
    "month_str", "stockout_rate", "overstock_rate", "fill_rate",
]].round(3).to_dict(orient="records"))

# Monthly revenue vs inventory (Spearman)
panel_mo = panel.groupby(panel["date"].dt.to_period("M").dt.to_timestamp()).agg(rev_month=("Revenue", "sum")).reset_index(names="month")
inv_merge = panel_mo.merge(inv_agg, on="month", how="inner")
if len(inv_merge) > 5:
    log("inventory.rho_stockout", float(stats.spearmanr(inv_merge["stockout_rate"], inv_merge["rev_month"]).statistic))
    log("inventory.rho_overstock", float(stats.spearmanr(inv_merge["overstock_rate"], inv_merge["rev_month"]).statistic))
    log("inventory.rho_fillrate", float(stats.spearmanr(inv_merge["fill_rate"], inv_merge["rev_month"]).statistic))

# --- Test G: Return reasons
reason_share = (returns["return_reason"].value_counts(normalize=True) * 100).round(1)
log("returns.reason_share_pct", reason_share.to_dict())
log("returns.n_total", int(returns.shape[0]))
refund_total = float(returns["refund_amount"].sum())
rev_total = float(panel["Revenue"].sum())
log("returns.refund_to_revenue_pct", refund_total / rev_total * 100)

# --- Test H: Customer concentration (Pareto)
cust_rev = oi.groupby("order_id")["net_item"].sum().reset_index().merge(
    orders[["order_id", "customer_id"]], on="order_id", how="left"
)
cust_tot = cust_rev.groupby("customer_id")["net_item"].sum().sort_values(ascending=False)
cum_share = cust_tot.cumsum() / cust_tot.sum()
top10 = float((cum_share.iloc[: int(0.1 * len(cust_tot))].iloc[-1]) * 100)
top20 = float((cum_share.iloc[: int(0.2 * len(cust_tot))].iloc[-1]) * 100)
order_count = orders.groupby("customer_id").size()
repeat_pct = float((order_count > 1).mean() * 100)
log("customers.n", int(cust_tot.shape[0]))
log("customers.top10_pct", top10)
log("customers.top20_pct", top20)
log("customers.repeat_pct", repeat_pct)
log("customers.orders_per_cust_mean", float(order_count.mean()))
log("customers.orders_per_cust_median", float(order_count.median()))

# Save metrics
(OUT / "metrics_v4.json").write_text(json.dumps(METRICS, indent=2, ensure_ascii=False))
print(f"  -> metrics_v4.json ({len(METRICS)} keys)")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------

print("[4/6] Rendering figures ...", flush=True)


def _fmt_vnd(x: float, _pos=None) -> str:
    if abs(x) >= 1e9:
        return f"{x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


# --- Fig 1: System overview (3 panels) ----------------------------------------
fig1, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

# Panel A: daily revenue + MA30 + MA365
ax = axes[0]
d = panel.set_index("date")["Revenue"]
ax.plot(d.index, d.values, color="#BBBBBB", linewidth=0.35, alpha=0.8, label="Daily")
ax.plot(d.index, d.rolling(30).mean(), color=OKABE_ITO[0], linewidth=1.0, label="30-d MA")
ax.plot(d.index, d.rolling(365).mean(), color=OKABE_ITO[1], linewidth=1.6, label="365-d MA")
ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"), color=OKABE_ITO[4], alpha=0.18, label="Regime break 2019")
ax.axvline(pd.Timestamp("2020-01-23"), color=OKABE_ITO[3], linestyle="--", linewidth=0.7, label="VN COVID case #1")
ax.set_title("A. Daily revenue 2012-2022")
ax.set_ylabel("Revenue (VND)")
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_fmt_vnd))
ax.legend(loc="upper right", frameon=False, fontsize=6)

# Panel B: quarterly + test horizon projection
ax = axes[1]
q = panel.set_index("date")["Revenue"].resample("QS").sum() / 1e6
ax.bar(q.index, q.values, width=80, color=OKABE_ITO[0], alpha=0.8)
# highlight 2022 vs earlier
for y, lbl in [(2016, "2016 peak"), (2022, "2022 rebound")]:
    yr = q[q.index.year == y]
    ax.bar(yr.index, yr.values, width=80, color=OKABE_ITO[1], alpha=0.95)
# Mark test horizon on x-axis
ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-07-01"), color=OKABE_ITO[2], alpha=0.22, label="Test horizon (548d)")
ax.set_title("B. Quarterly revenue + test window")
ax.set_ylabel("Revenue (M VND / quarter)")
ax.legend(loc="upper right", frameon=False, fontsize=6)

# Panel C: regime break distributions (pre vs post 2019)
ax = axes[2]
pre_m = pre / 1e6
post_m = post / 1e6
bins = np.linspace(0, max(pre_m.max(), post_m.max()) * 1.02, 45)
ax.hist(pre_m, bins=bins, alpha=0.55, color=OKABE_ITO[0], label=f"2012-2018 (n={len(pre)})", density=True)
ax.hist(post_m, bins=bins, alpha=0.55, color=OKABE_ITO[1], label=f"2019-2022 (n={len(post)})", density=True)
ax.axvline(np.mean(pre_m), color=OKABE_ITO[0], linestyle="--", linewidth=0.8)
ax.axvline(np.mean(post_m), color=OKABE_ITO[1], linestyle="--", linewidth=0.8)
ax.set_xlabel("Daily Revenue (M VND)")
ax.set_ylabel("Density")
ax.set_title(f"C. Pre vs post-2019 (t={t_stat:.1f})")
ax.legend(frameon=False, fontsize=6)

fig1.tight_layout()
_save(fig1, "fig1_system_overview")


# --- Fig 2: Seasonality (2 panels) --------------------------------------------
fig2, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1.6]})

# Panel A: monthly mean bar with 95% CI
ax = axes[0]
month_groups = [panel[panel["month"] == m]["Revenue"] / 1e6 for m in range(1, 13)]
means = np.array([g.mean() for g in month_groups])
sems = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in month_groups])
colors = np.where(means >= means.mean(), OKABE_ITO[2], OKABE_ITO[4])
ax.bar(range(1, 13), means, yerr=sems * 1.96, color=colors, capsize=2.5, linewidth=0)
ax.axhline(means.mean(), color="black", linewidth=0.6, linestyle=":", label="Grand mean")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month")
ax.set_ylabel("Daily revenue (M VND, mean \u00b195% CI)")
ax.set_title(f"A. Q2 peak (May +{month_delta_pct[5]:.0f}%) vs Dec trough ({month_delta_pct[12]:.0f}%)")
ax.legend(frameon=False, fontsize=6)

# Panel B: month x dow heatmap of % deviation
ax = axes[1]
im = ax.imshow(mdow_pct.values, aspect="auto", cmap="RdBu_r",
               vmin=-max(abs(mdow_pct.values.min()), abs(mdow_pct.values.max())),
               vmax=max(abs(mdow_pct.values.min()), abs(mdow_pct.values.max())))
ax.set_xticks(range(12))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_yticks(range(7))
ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xlabel("Month")
ax.set_ylabel("Day of week")
ax.set_title("B. Month \u00d7 DoW revenue % vs grand mean")
cbar = fig2.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("% dev.", fontsize=7)
cbar.ax.tick_params(labelsize=6)
# annotate extremes
mx = np.unravel_index(np.argmax(mdow_pct.values), mdow_pct.shape)
mn = np.unravel_index(np.argmin(mdow_pct.values), mdow_pct.shape)
ax.text(mx[1], mx[0], f"+{mdow_pct.values[mx]:.0f}%", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
ax.text(mn[1], mn[0], f"{mdow_pct.values[mn]:.0f}%", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
ax.grid(False)

fig2.tight_layout()
_save(fig2, "fig2_seasonality")


# --- Fig 3: Structural risks (2 panels) ---------------------------------------
fig3, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1.3]})

# Panel A: category revenue share (stacked horizontal bar + HHI)
ax = axes[0]
cat_sorted = cat_share.sort_values(ascending=True)
colors_cat = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(cat_sorted))]
ypos = np.arange(len(cat_sorted))
ax.barh(ypos, cat_sorted.values * 100, color=colors_cat)
for y, v in zip(ypos, cat_sorted.values):
    ax.text(v * 100 + 0.8, y, f"{v*100:.1f}%", va="center", fontsize=7)
ax.set_yticks(ypos)
ax.set_yticklabels(cat_sorted.index)
ax.set_xlabel("Revenue share (%)")
ax.set_title(f"A. Category concentration (HHI = {hhi_cat:.2f})")
ax.set_xlim(0, max(cat_sorted.values) * 100 * 1.2)

# Panel B: promo bucket box + Mann-Whitney annotation
ax = axes[1]
bucket_data = [panel.loc[panel["promo_bucket"] == lbl, "Revenue"].values / 1e6 for lbl in labels]
bp = ax.boxplot(bucket_data, labels=labels, patch_artist=True, widths=0.55, showfliers=False)
for patch, c in zip(bp["boxes"], [OKABE_ITO[0], OKABE_ITO[5], OKABE_ITO[4], OKABE_ITO[1]]):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for med in bp["medians"]:
    med.set_color("black")
    med.set_linewidth(1.2)
ax.set_xlabel("% orders using promo that day")
ax.set_ylabel("Daily revenue (M VND)")
ax.set_title(f"B. Heavy vs light promo: median \u2212{gap_pct:.1f}%")

# annotate n per bucket
counts = [len(b) for b in bucket_data]
for i, c in enumerate(counts, start=1):
    ax.text(i, ax.get_ylim()[0] + 0.5, f"n={c}", ha="center", fontsize=6, color="#555555")

fig3.tight_layout()
_save(fig3, "fig3_structural")


# --- Fig 4: Operational diagnosis (2 panels) ----------------------------------
fig4, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1.3]})

# Panel A: stockout vs overstock scatter (inventory paradox)
ax = axes[0]
inv_last24 = inv_agg[inv_agg["month"] >= "2021-01-01"].copy()
inv_last24["year"] = inv_last24["month"].dt.year
for y, c in zip(sorted(inv_last24["year"].unique()), [OKABE_ITO[0], OKABE_ITO[1]]):
    sub = inv_last24[inv_last24["year"] == y]
    ax.scatter(sub["stockout_rate"] * 100, sub["overstock_rate"] * 100,
               color=c, s=55, alpha=0.85, edgecolor="black", linewidth=0.4, label=str(y))
ax.axhline(50, color="grey", linewidth=0.4, linestyle=":")
ax.axvline(50, color="grey", linewidth=0.4, linestyle=":")
ax.set_xlabel("Stockout rate (% of SKUs)")
ax.set_ylabel("Overstock rate (% of SKUs)")
ax.set_title("A. Paradox: SKUs both stock-out & over-stock")
ax.legend(frameon=False, fontsize=6, loc="lower right", title="Year")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Panel B: exogenous signal ranking (leakage-tagged)
ax = axes[1]
r_items = sorted(ranks.items(), key=lambda kv: abs(kv[1]), reverse=True)
# Identify leakage risks (same-day COGS / n_orders / refund)
leakage = {"COGS", "n_orders", "refund", "margin"}
names = [k for k, _ in r_items]
vals = [v for _, v in r_items]
colors_bar = [OKABE_ITO[1] if n in leakage else OKABE_ITO[0] for n in names]
ypos = np.arange(len(names))
ax.barh(ypos, vals, color=colors_bar)
ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=6)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.4)
ax.set_xlabel("Spearman \u03c1 with Revenue (same-day)")
ax.set_title("B. Exogenous ranking (orange = leakage risk)")
for y, v in zip(ypos, vals):
    ax.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=6)
ax.set_xlim(-1.05, 1.1)

fig4.tight_layout()
_save(fig4, "fig4_operations")


# --- Fig 5: Forecast readiness (CV timeline) ---------------------------------
fig5, ax = plt.subplots(1, 1, figsize=(7.4, 2.0))

start = pd.Timestamp("2012-07-04")
train_end = pd.Timestamp("2022-12-31")
test_start = pd.Timestamp("2023-01-01")
test_end = pd.Timestamp("2024-07-01")

# Folds (rolling-origin, each validates on 548 days)
folds = [
    ("Fold 1", pd.Timestamp("2012-07-04"), pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2021-12-31")),
    ("Fold 2", pd.Timestamp("2012-07-04"), pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")),
]
# background
ax.axvspan(start, train_end, color="#EEEEEE", alpha=0.9)
ax.axvspan(test_start, test_end, color=OKABE_ITO[2], alpha=0.32, label="Test (548d, unseen)")
ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"), color=OKABE_ITO[4], alpha=0.22, label="Regime break 2019")

y_positions = [0.65, 0.35]
for (name, ts, te, vs, ve), y in zip(folds, y_positions):
    train_lbl = "Train window" if y == y_positions[0] else None
    val_lbl = "Validation (548d)" if y == y_positions[0] else None
    ax.hlines(y, ts, te, colors=OKABE_ITO[0], linewidth=8, label=train_lbl)
    ax.hlines(y, vs, ve, colors=OKABE_ITO[1], linewidth=8, label=val_lbl)
    ax.text(ts - pd.Timedelta(days=80), y, name, fontsize=7, fontweight="bold",
            ha="right", va="center")

ax.set_ylim(0.1, 0.95)
ax.set_yticks([])
ax.set_xlim(start - pd.Timedelta(days=300), test_end + pd.Timedelta(days=60))
ax.set_title("Rolling-origin CV: 2 folds \u00d7 548-day horizon, matched to unseen test window")
ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
ax.grid(False)

fig5.tight_layout()
_save(fig5, "fig5_forecast_readiness")


print("[5/6] Saving metrics ...", flush=True)
# Rewrite metrics (figures may have added auxiliary data)
(OUT / "metrics_v4.json").write_text(json.dumps(METRICS, indent=2, ensure_ascii=False))

print("[6/6] Done. Outputs:")
for p in sorted(OUT.glob("**/*")):
    if p.is_file():
        print(f"  - {p.relative_to(ROOT)}")
