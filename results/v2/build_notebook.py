"""Helper: build results/v2/eda.ipynb from a list of code/markdown cells.

Run with: uv run python results/v2/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "eda.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS: list[dict] = []


# ---------------------------------------------------------------------------
# 0. Title + context
# ---------------------------------------------------------------------------

CELLS.append(md(
    """# EDA v2 — Datathon 2026 *The Gridbreakers*

Đội: Data Science Team · Ngày build: 2026-04-17

Notebook này được nâng cấp từ `results/v1/eda.ipynb` theo hướng **system-thinking** (xem `evaluation_citeria.md`): mỗi phân tích đi theo pipeline **visualize → quantify → validate → interpret → forecasting-implication**.

Mục tiêu kinh doanh: dự báo `Revenue` hàng ngày cho `2023-01-01 → 2024-07-01` (548 ngày).
Train range: `2012-07-04 → 2022-12-31` (3,833 ngày).

Quy ước lưu ảnh:
- Ảnh EDA (diagnostic, mô tả dữ liệu): `results/v2/images/eda/<NN>_<slug>.png`
- Ảnh báo cáo (insight-focused): `results/v2/images/report/<slug>.png`

Các metric quan trọng được append vào `results/v2/metrics.json` để báo cáo (`report.md`) trích dẫn số liệu thật, không hallucinate.
"""
))


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

CELLS.append(md("## 1. Setup, load & sanity checks"))

CELLS.append(code(
    """from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats

warnings.filterwarnings(\"ignore\")

sns.set_theme(style=\"whitegrid\", context=\"notebook\")
plt.rcParams[\"figure.dpi\"] = 110
plt.rcParams[\"savefig.dpi\"] = 140
plt.rcParams[\"savefig.bbox\"] = \"tight\"

ROOT = Path.cwd()
while not (ROOT / \"data\").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / \"data\"
OUT = ROOT / \"results\" / \"v2\"
IMG_EDA = OUT / \"images\" / \"eda\"
IMG_REP = OUT / \"images\" / \"report\"
IMG_EDA.mkdir(parents=True, exist_ok=True)
IMG_REP.mkdir(parents=True, exist_ok=True)

METRICS: dict = {}

def log(key: str, value) -> None:
    \"\"\"Append a metric to METRICS (json-safe).\"\"\"
    def to_safe(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        if isinstance(v, dict):
            return {str(k): to_safe(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [to_safe(x) for x in v]
        if isinstance(v, pd.DataFrame):
            return v.round(4).to_dict(orient=\"records\")
        if isinstance(v, pd.Series):
            return {str(k): to_safe(val) for k, val in v.items()}
        return v
    METRICS[key] = to_safe(value)

def save_eda(name: str) -> Path:
    plt.tight_layout()
    path = IMG_EDA / name
    plt.savefig(path)
    plt.show()
    return path

def save_report(name: str) -> Path:
    plt.tight_layout()
    path = IMG_REP / name
    plt.savefig(path)
    plt.show()
    return path

print(\"DATA:\", DATA)
print(\"OUT :\", OUT)
"""
))

CELLS.append(code(
    """DATE_COLS = {
    \"orders\": [\"order_date\"],
    \"order_items\": [],
    \"customers\": [\"signup_date\"],
    \"products\": [],
    \"promotions\": [\"start_date\", \"end_date\"],
    \"payments\": [],
    \"shipments\": [\"ship_date\", \"delivery_date\"],
    \"returns\": [\"return_date\"],
    \"reviews\": [\"review_date\"],
    \"geography\": [],
    \"inventory\": [\"snapshot_date\"],
    \"web_traffic\": [\"date\"],
    \"sales\": [\"Date\"],
    \"sample_submission\": [\"Date\"],
}

def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f\"{name}.csv\", parse_dates=DATE_COLS[name])
    return df

sales = load(\"sales\")
orders = load(\"orders\")
order_items = load(\"order_items\")
products = load(\"products\")
customers = load(\"customers\")
promotions = load(\"promotions\")
payments = load(\"payments\")
shipments = load(\"shipments\")
returns = load(\"returns\")
reviews = load(\"reviews\")
geography = load(\"geography\")
inventory = load(\"inventory\")
web = load(\"web_traffic\")
sample_sub = load(\"sample_submission\")

TABLES = {
    \"sales\": sales, \"orders\": orders, \"order_items\": order_items,
    \"products\": products, \"customers\": customers, \"promotions\": promotions,
    \"payments\": payments, \"shipments\": shipments, \"returns\": returns,
    \"reviews\": reviews, \"geography\": geography, \"inventory\": inventory,
    \"web_traffic\": web, \"sample_submission\": sample_sub,
}
for name, df in TABLES.items():
    print(f\"{name:18s} {df.shape}\")
"""
))

CELLS.append(md("### 1.1 Shape, missingness, date coverage"))

CELLS.append(code(
    """shape_rows = []
for name, df in TABLES.items():
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    dmin = min((df[c].min() for c in dt_cols), default=pd.NaT)
    dmax = max((df[c].max() for c in dt_cols), default=pd.NaT)
    miss = df.isna().mean().max() if len(df.columns) else 0
    shape_rows.append({
        \"table\": name,
        \"rows\": len(df),
        \"cols\": df.shape[1],
        \"date_min\": dmin,
        \"date_max\": dmax,
        \"max_col_missing_%\": round(miss * 100, 2),
    })
shape_df = pd.DataFrame(shape_rows).set_index(\"table\")
display(shape_df)
log(\"shape_summary\", shape_df.reset_index())
"""
))

CELLS.append(code(
    """miss = pd.DataFrame({
    name: df.isna().mean() for name, df in TABLES.items()
}).T.fillna(0)
miss_significant = miss.loc[:, (miss > 0.001).any(axis=0)]
plt.figure(figsize=(12, 5))
sns.heatmap(
    miss_significant * 100,
    cmap=\"Reds\",
    annot=True,
    fmt=\".1f\",
    cbar_kws={\"label\": \"% missing\"},
    linewidths=0.3,
    linecolor=\"white\",
)
plt.title(\"Tỉ lệ missing (%) theo bảng × cột (chỉ cột có missing)\")
plt.xlabel(\"\"); plt.ylabel(\"\")
save_eda(\"01_missingness_heatmap.png\")

log(\"missingness_top\", {
    f\"{tbl}.{col}\": float(miss.loc[tbl, col] * 100)
    for tbl in miss.index
    for col in miss.columns
    if miss.loc[tbl, col] > 0.001
})
"""
))

CELLS.append(code(
    """# Date coverage per day — detect any gap in daily series
for name, col in [(\"sales\", \"Date\"), (\"orders\", \"order_date\"), (\"web_traffic\", \"date\")]:
    df = TABLES[name]
    uniq_days = df[col].dt.normalize().nunique()
    span = (df[col].max() - df[col].min()).days + 1
    gap = span - uniq_days
    print(f\"{name:12s}  unique days={uniq_days:5d}  span_days={span:5d}  gaps={gap}\")
    log(f\"coverage_{name}\", {\"unique_days\": int(uniq_days), \"span_days\": int(span), \"gaps\": int(gap)})
"""
))

CELLS.append(code(
    """# Basic referential integrity checks (counts only)
checks = {
    \"order_items→orders\": (~order_items.order_id.isin(orders.order_id)).sum(),
    \"order_items→products\": (~order_items.product_id.isin(products.product_id)).sum(),
    \"orders→customers\": (~orders.customer_id.isin(customers.customer_id)).sum(),
    \"orders→geography(zip)\": (~orders.zip.isin(geography.zip)).sum(),
    \"payments→orders\": (~payments.order_id.isin(orders.order_id)).sum(),
    \"shipments→orders\": (~shipments.order_id.isin(orders.order_id)).sum(),
    \"returns→orders\": (~returns.order_id.isin(orders.order_id)).sum(),
    \"reviews→orders\": (~reviews.order_id.isin(orders.order_id)).sum(),
    \"reviews→customers\": (~reviews.customer_id.isin(customers.customer_id)).sum(),
    \"inventory→products\": (~inventory.product_id.isin(products.product_id)).sum(),
}
ri = pd.Series(checks, name=\"orphan_rows\")
display(ri.to_frame())
log(\"referential_integrity\", ri)
"""
))


# ---------------------------------------------------------------------------
# 2. Revenue history + regime
# ---------------------------------------------------------------------------

CELLS.append(md(
    """## 2. Lịch sử Revenue & phát hiện regime change

Mục tiêu forecast là `Revenue` hàng ngày. Trước khi nói về mùa vụ, ta phải hỏi: *mẫu của 10 năm quá khứ có đồng nhất không?* Nếu không, train toàn bộ với trọng số đều sẽ sai.
"""
))

CELLS.append(code(
    """sales = sales.sort_values(\"Date\").reset_index(drop=True)
sales[\"margin\"] = (sales.Revenue - sales.COGS) / sales.Revenue
sales[\"year\"] = sales.Date.dt.year
sales[\"month\"] = sales.Date.dt.month
sales[\"quarter\"] = sales.Date.dt.to_period(\"Q\").astype(str)
sales[\"dow\"] = sales.Date.dt.day_name()
sales[\"ma30\"] = sales.Revenue.rolling(30, min_periods=5).mean()
sales[\"ma90\"] = sales.Revenue.rolling(90, min_periods=30).mean()
sales[\"ma365\"] = sales.Revenue.rolling(365, min_periods=60).mean()
display(sales.head())
display(sales[[\"Revenue\", \"COGS\", \"margin\"]].describe())
"""
))

CELLS.append(code(
    """fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sales.Date, sales.Revenue / 1e6, color=\"#cfd8dc\", linewidth=0.6, label=\"Daily\")
ax.plot(sales.Date, sales.ma30 / 1e6, color=\"#1565c0\", linewidth=1.2, label=\"MA30\")
ax.plot(sales.Date, sales.ma365 / 1e6, color=\"#c62828\", linewidth=2.0, label=\"MA365\")
ax.set_title(\"Doanh thu hàng ngày 2012–2022 (triệu VND) + đường trung bình trượt\")
ax.set_ylabel(\"Revenue (triệu VND)\")
ax.set_xlabel(\"\")
ax.legend()
save_eda(\"02_revenue_history_ma.png\")
"""
))

CELLS.append(code(
    """yearly = sales.groupby(\"year\").agg(
    revenue=(\"Revenue\", \"sum\"),
    cogs=(\"COGS\", \"sum\"),
    days=(\"Date\", \"nunique\"),
)
yearly[\"gross_margin\"] = (yearly.revenue - yearly.cogs) / yearly.revenue
yearly[\"yoy_pct\"] = yearly.revenue.pct_change() * 100
display(yearly)
log(\"yearly_revenue\", yearly.reset_index())
"""
))

CELLS.append(code(
    """# Structural break test: t-test on daily revenue 2012–2018 vs 2019–2022
pre = sales.loc[sales.year.between(2012, 2018), \"Revenue\"]
post = sales.loc[sales.year.between(2019, 2022), \"Revenue\"]
t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
cohen_d = (pre.mean() - post.mean()) / np.sqrt((pre.var() + post.var()) / 2)
print(f\"pre mean  = {pre.mean():,.0f}\")
print(f\"post mean = {post.mean():,.0f}\")
print(f\"drop      = {(1 - post.mean()/pre.mean())*100:.1f}%\")
print(f\"t-stat={t_stat:.2f}, p={p_val:.2e}, Cohen d={cohen_d:.2f}\")
log(\"regime_test\", {
    \"pre_mean\": pre.mean(),
    \"post_mean\": post.mean(),
    \"drop_pct\": (1 - post.mean()/pre.mean())*100,
    \"t_stat\": float(t_stat),
    \"p_value\": float(p_val),
    \"cohen_d\": float(cohen_d),
})
"""
))

CELLS.append(code(
    """# Day-over-day change detection: which month shows the largest negative jump
monthly = sales.set_index(\"Date\").Revenue.resample(\"MS\").sum().rename(\"rev\")
monthly_log_diff = np.log(monthly).diff()
top_negatives = monthly_log_diff.nsmallest(6)
display(top_negatives.to_frame(\"log_diff\"))
log(\"monthly_biggest_drops\", top_negatives)
"""
))

CELLS.append(code(
    """# Report chart: annotated trend — MA30/MA365 with regime split
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(sales.Date, sales.ma30 / 1e6, color=\"#1565c0\", linewidth=1.0, label=\"MA30\")
ax.plot(sales.Date, sales.ma365 / 1e6, color=\"#b71c1c\", linewidth=2.2, label=\"MA365\")
ax.axvspan(pd.Timestamp(\"2019-01-01\"), pd.Timestamp(\"2019-12-31\"),
           alpha=0.12, color=\"#ef6c00\", label=\"Regime break 2019\")
ax.axvline(pd.Timestamp(\"2020-01-23\"), color=\"#6a1b9a\", linestyle=\"--\", alpha=0.7, label=\"COVID-19 VN\")
ax.set_ylabel(\"Revenue (triệu VND)\")
ax.set_xlabel(\"\")
ax.set_title(\"Regime change trước COVID — doanh thu sụp mạnh ngay trong 2019\")
ax.legend(loc=\"upper right\")
save_report(\"revenue_regime_change.png\")
"""
))


# ---------------------------------------------------------------------------
# 3. Seasonality
# ---------------------------------------------------------------------------

CELLS.append(md("## 3. Mùa vụ: tháng × weekday × quý"))

CELLS.append(code(
    """monthly_mean = sales.groupby(\"month\").Revenue.mean()
dow_order = [\"Monday\", \"Tuesday\", \"Wednesday\", \"Thursday\", \"Friday\", \"Saturday\", \"Sunday\"]
dow_mean = sales.groupby(\"dow\").Revenue.mean().reindex(dow_order)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
monthly_mean.plot(kind=\"bar\", ax=axes[0], color=\"#1565c0\")
axes[0].set_title(\"Doanh thu TB theo tháng\")
axes[0].set_xlabel(\"Tháng\"); axes[0].set_ylabel(\"Revenue TB\")
dow_mean.plot(kind=\"bar\", ax=axes[1], color=\"#2e7d32\")
axes[1].set_title(\"Doanh thu TB theo thứ\")
axes[1].set_xlabel(\"\"); axes[1].set_ylabel(\"\")
plt.xticks(rotation=30)
save_eda(\"03_monthly_weekday_mean.png\")
log(\"monthly_mean\", monthly_mean)
log(\"dow_mean\", dow_mean)
"""
))

CELLS.append(code(
    """# Seasonal heatmap (dow × month) as % deviation from overall mean
overall = sales.Revenue.mean()
heat = sales.groupby([\"month\", \"dow\"]).Revenue.mean().unstack()[dow_order]
heat_pct = (heat - overall) / overall * 100

plt.figure(figsize=(11, 5.5))
sns.heatmap(heat_pct, cmap=\"RdBu_r\", center=0, annot=True, fmt=\".0f\",
            cbar_kws={\"label\": \"% lệch mean ngày\"}, linewidths=0.3)
plt.title(\"Mùa vụ: % lệch Revenue TB so với mean 10 năm\")
plt.xlabel(\"\"); plt.ylabel(\"Tháng\")
save_report(\"seasonality_month_dow_heatmap.png\")

flat = heat_pct.stack().sort_values()
log(\"seasonality_bottom5\", flat.head(5))
log(\"seasonality_top5\", flat.tail(5))
display(pd.concat([flat.head(5).rename(\"% lệch mean\"), flat.tail(5).rename(\"% lệch mean\")]).to_frame())
"""
))

CELLS.append(code(
    """# Quarterly view — for forecasting we care about Q-level seasonality too
qtr = sales.groupby(pd.Grouper(key=\"Date\", freq=\"QS\")).Revenue.sum() / 1e9
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(np.arange(len(qtr)), qtr.values, color=\"#37474f\")
ax.set_xticks(np.arange(len(qtr))[::2])
ax.set_xticklabels([pd.Timestamp(d).strftime(\"%Y-Q%q\").replace(\"-Q1\", \"-Q1\")
                    if False else f\"{pd.Timestamp(d).year}Q{((pd.Timestamp(d).month-1)//3)+1}\"
                    for d in qtr.index[::2]], rotation=45)
ax.set_title(\"Doanh thu theo quý (tỷ VND)\")
ax.set_ylabel(\"Revenue (tỷ VND)\")
save_eda(\"04_quarterly_revenue.png\")
log(\"quarterly_revenue_bn\", {f\"{pd.Timestamp(d).year}Q{((pd.Timestamp(d).month-1)//3)+1}\": float(v) for d, v in qtr.items()})
"""
))


# ---------------------------------------------------------------------------
# 4. Revenue vs COGS / margin
# ---------------------------------------------------------------------------

CELLS.append(md("## 4. Revenue vs COGS & tỷ suất lợi nhuận"))

CELLS.append(code(
    """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(sales.COGS / 1e6, sales.Revenue / 1e6, s=5, alpha=0.3, color=\"#1565c0\")
axes[0].plot([0, sales.COGS.max()/1e6], [0, sales.COGS.max()/1e6],
             linestyle=\"--\", color=\"grey\", label=\"y=x\")
axes[0].set_xlabel(\"COGS (triệu VND)\"); axes[0].set_ylabel(\"Revenue (triệu VND)\")
axes[0].set_title(\"Revenue ≈ k · COGS\")
axes[0].legend()
sns.histplot(sales.margin, bins=60, ax=axes[1], color=\"#e65100\")
axes[1].axvline(sales.margin.mean(), color=\"red\", linestyle=\"--\", label=f\"mean={sales.margin.mean():.2%}\")
axes[1].set_xlabel(\"Margin hàng ngày\")
axes[1].set_title(\"Phân phối margin\")
axes[1].legend()
save_eda(\"05_revenue_vs_cogs_margin.png\")
log(\"margin_stats\", {
    \"mean\": sales.margin.mean(),
    \"median\": sales.margin.median(),
    \"std\": sales.margin.std(),
    \"p05\": sales.margin.quantile(0.05),
    \"p95\": sales.margin.quantile(0.95),
})
"""
))

CELLS.append(code(
    """# Regress Revenue on COGS — slope ~1/(1-margin)
slope, intercept, r, p, se = stats.linregress(sales.COGS, sales.Revenue)
print(f\"Revenue = {slope:.3f} * COGS + {intercept:,.0f}\")
print(f\"R²={r**2:.4f}  p~0   se={se:.4f}\")
log(\"revenue_vs_cogs_regression\", {
    \"slope\": float(slope), \"intercept\": float(intercept),
    \"r2\": float(r**2), \"se\": float(se),
})
"""
))

CELLS.append(code(
    """# Margin drift over time (90d rolling) — useful exogenous signal
sales[\"margin_ma90\"] = sales.margin.rolling(90, min_periods=30).mean()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(sales.Date, sales.margin_ma90 * 100, color=\"#6a1b9a\", linewidth=1.8)
ax.set_title(\"Gross margin 90d rolling — biến động theo thời gian\")
ax.set_ylabel(\"Margin (%)\")
ax.set_xlabel(\"\")
save_eda(\"06_margin_rolling.png\")
"""
))


# ---------------------------------------------------------------------------
# 5. Orders & customer behavior
# ---------------------------------------------------------------------------

CELLS.append(md("## 5. Đơn hàng & hành vi khách hàng"))

CELLS.append(code(
    """orders[\"order_day\"] = orders.order_date.dt.normalize()
orders[\"year\"] = orders.order_date.dt.year
daily_orders = orders.groupby(\"order_day\").size().rename(\"n_orders\")
daily_orders.index = pd.to_datetime(daily_orders.index)

merged = sales.set_index(\"Date\").join(daily_orders, how=\"left\")
merged[\"n_orders\"] = merged.n_orders.fillna(0)
corr_ro = merged[[\"Revenue\", \"n_orders\"]].corr(\"spearman\").iloc[0, 1]
print(f\"Spearman(Revenue, #orders) = {corr_ro:.3f}\")

aov = merged.Revenue / merged.n_orders.replace(0, np.nan)
aov_median = aov.median()
print(f\"AOV median (Revenue/#orders) ~ {aov_median:,.0f} VND\")
log(\"rev_vs_orders\", {\"spearman\": float(corr_ro), \"aov_median\": float(aov_median)})
"""
))

CELLS.append(code(
    """# Order status distribution
status_pct = orders.order_status.value_counts(normalize=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
status_pct.plot(kind=\"bar\", ax=axes[0], color=\"#2e7d32\")
axes[0].set_title(\"Phân phối order_status (%)\")
axes[0].set_ylabel(\"%\"); axes[0].tick_params(axis=\"x\", rotation=30)

(orders.device_type.value_counts(normalize=True) * 100).plot(
    kind=\"bar\", ax=axes[1], color=\"#00695c\")
axes[1].set_title(\"Phân phối device_type (%)\")
axes[1].set_ylabel(\"%\"); axes[1].tick_params(axis=\"x\", rotation=30)
save_eda(\"07_orders_status_device.png\")

log(\"order_status_pct\", status_pct)
log(\"order_device_pct\", orders.device_type.value_counts(normalize=True) * 100)
log(\"order_source_pct\", orders.order_source.value_counts(normalize=True) * 100)
"""
))

CELLS.append(code(
    """# Customer-level: Pareto + repeat purchase
cust_orders = orders.groupby(\"customer_id\").size().sort_values(ascending=False)
cust_revenue = (
    order_items.assign(line_rev=lambda d: d.unit_price * d.quantity - d.discount_amount)
    .merge(orders[[\"order_id\", \"customer_id\"]], on=\"order_id\", how=\"left\")
    .groupby(\"customer_id\").line_rev.sum()
    .sort_values(ascending=False)
)
cum = cust_revenue.cumsum() / cust_revenue.sum()

thresholds = [0.1, 0.2, 0.4, 0.8]
pareto = {}
n_cust = len(cust_revenue)
for t in thresholds:
    n = int(np.ceil(n_cust * t))
    pareto[f\"Top {int(t*100)}%\"] = {\n        \"n_customers\": n,\n        \"cum_revenue_share\": float(cum.iloc[n-1]),\n    }
repeat_rate = (cust_orders > 1).mean()
print(\"Pareto:\")
for k, v in pareto.items():
    print(f\"  {k}: {v['n_customers']:,} khách → {v['cum_revenue_share']*100:.1f}% doanh thu\")
print(f\"Repeat-rate (>1 đơn): {repeat_rate*100:.1f}%\")
print(f\"Orders/customer  mean={cust_orders.mean():.2f}, median={cust_orders.median():.1f}\")
log(\"pareto\", pareto)
log(\"repeat_rate\", float(repeat_rate))
log(\"orders_per_customer\", {\"mean\": float(cust_orders.mean()), \"median\": float(cust_orders.median())})
"""
))

CELLS.append(code(
    """# Active customers — important forecasting feature
orders[\"month_start\"] = orders.order_date.values.astype(\"datetime64[M]\")
active_monthly = orders.groupby(\"month_start\").customer_id.nunique()
new_cust = customers.assign(m=customers.signup_date.values.astype(\"datetime64[M]\")).groupby(\"m\").size()

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(active_monthly.index, active_monthly.values, color=\"#1565c0\", linewidth=1.8, label=\"Active customers / tháng\")
ax.plot(new_cust.index, new_cust.values, color=\"#c62828\", linewidth=1.4, label=\"New signups / tháng\")
ax.set_title(\"Khách active vs khách mới theo tháng\")
ax.set_ylabel(\"Số khách\")
ax.legend()
save_eda(\"08_active_vs_new_customers.png\")

log(\"active_customers_last12m_mean\", float(active_monthly.tail(12).mean()))
log(\"new_customers_last12m_mean\", float(new_cust.tail(12).mean()))
"""
))


# ---------------------------------------------------------------------------
# 6. Product / category / segment mix
# ---------------------------------------------------------------------------

CELLS.append(md("## 6. Product / Category / Segment mix"))

CELLS.append(code(
    """oi_enriched = order_items.merge(
    products[[\"product_id\", \"category\", \"segment\"]], on=\"product_id\", how=\"left\"
).merge(orders[[\"order_id\", \"order_date\"]], on=\"order_id\", how=\"left\")
oi_enriched[\"line_rev\"] = oi_enriched.unit_price * oi_enriched.quantity - oi_enriched.discount_amount

cat_rev = oi_enriched.groupby(\"category\").line_rev.sum().sort_values(ascending=False) / 1e9
seg_rev = oi_enriched.groupby(\"segment\").line_rev.sum().sort_values(ascending=False) / 1e9

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
cat_rev.plot(kind=\"bar\", ax=axes[0], color=\"#1565c0\")
axes[0].set_title(\"Doanh thu theo category (tỷ VND, 2012–2022)\")
axes[0].set_ylabel(\"Revenue (tỷ VND)\")
axes[0].tick_params(axis=\"x\", rotation=20)
seg_rev.plot(kind=\"bar\", ax=axes[1], color=\"#ef6c00\")
axes[1].set_title(\"Doanh thu theo segment (tỷ VND)\")
axes[1].tick_params(axis=\"x\", rotation=20)
save_eda(\"09_revenue_by_category_segment.png\")
log(\"cat_revenue_bn\", cat_rev)
log(\"seg_revenue_bn\", seg_rev)
"""
))

CELLS.append(code(
    """# Concentration of revenue across categories (HHI-like)
cat_share = cat_rev / cat_rev.sum()
hhi = (cat_share ** 2).sum()
print(f\"Category HHI = {hhi:.3f} (1.0 = độc quyền, 1/k = cân bằng hoàn hảo, k=4 → 0.25)\")
display(cat_share.to_frame(\"share\"))
log(\"category_hhi\", float(hhi))
log(\"category_share\", cat_share)
"""
))

CELLS.append(code(
    """# Category seasonality — do all cats share same monthly pattern?
oi_enriched[\"month\"] = oi_enriched.order_date.dt.month
cat_month = oi_enriched.groupby([\"month\", \"category\"]).line_rev.sum().unstack()
cat_month_pct = cat_month.div(cat_month.sum(axis=0), axis=1) * 100

plt.figure(figsize=(11, 5))
sns.heatmap(cat_month_pct, annot=True, fmt=\".1f\", cmap=\"viridis\",
            cbar_kws={\"label\": \"% doanh thu cả năm\"})
plt.title(\"Mùa vụ theo category (% doanh thu phân bổ cho mỗi tháng)\")
plt.xlabel(\"Category\"); plt.ylabel(\"Tháng\")
save_report(\"category_monthly_seasonality.png\")
log(\"category_month_share\", cat_month_pct)
"""
))


# ---------------------------------------------------------------------------
# 7. Promotion behavior
# ---------------------------------------------------------------------------

CELLS.append(md("## 7. Hành vi khuyến mãi"))

CELLS.append(code(
    """# Build daily promo signal
oi_with_date = order_items.merge(orders[[\"order_id\", \"order_date\"]], on=\"order_id\")
oi_with_date[\"has_promo\"] = oi_with_date.promo_id.notna() | oi_with_date.promo_id_2.notna()
oi_with_date[\"order_day\"] = oi_with_date.order_date.dt.normalize()

daily_promo = oi_with_date.groupby(\"order_day\").agg(
    promo_lines=(\"has_promo\", \"sum\"),
    total_lines=(\"has_promo\", \"count\"),
    total_discount=(\"discount_amount\", \"sum\"),
)
daily_promo[\"promo_share\"] = daily_promo.promo_lines / daily_promo.total_lines
daily_promo.index = pd.to_datetime(daily_promo.index)
merged_promo = sales.set_index(\"Date\").join(daily_promo, how=\"left\").fillna(0)

merged_promo[\"promo_bucket\"] = pd.cut(
    merged_promo.promo_share,
    bins=[-0.01, 0.10, 0.25, 0.50, 1.01],
    labels=[\"<10%\", \"10–25%\", \"25–50%\", \">50%\"],
)
bucket_stats = merged_promo.groupby(\"promo_bucket\").Revenue.agg([\"count\", \"mean\", \"median\"])
display(bucket_stats)
log(\"promo_bucket_stats\", bucket_stats.reset_index())
"""
))

CELLS.append(code(
    """# Non-parametric test — does heavy-promo day have a different revenue distribution?
light = merged_promo.loc[merged_promo.promo_bucket == \"<10%\", \"Revenue\"]
heavy = merged_promo.loc[merged_promo.promo_bucket == \">50%\", \"Revenue\"]
u_stat, p_val = stats.mannwhitneyu(light, heavy, alternative=\"two-sided\")
d = (light.median() - heavy.median()) / light.median() * 100  # heavy lower than light by d%, baseline = light
print(f\"Median light={light.median():,.0f}  heavy={heavy.median():,.0f}\")
print(f\"Heavy - light median = {heavy.median() - light.median():,.0f} ({-d:.1f}% thấp hơn)\")
print(f\"Mann-Whitney U={u_stat:.0f}  p={p_val:.2e}\")
log(\"promo_mwu\", {\n    \"light_median\": float(light.median()), \n    \"heavy_median\": float(heavy.median()),\n    \"diff_pct\": float(-d), \n    \"u_stat\": float(u_stat), \n    \"p_value\": float(p_val),\n})
"""
))

CELLS.append(code(
    """plot_df = merged_promo[[\"promo_bucket\", \"Revenue\"]].copy()
plot_df[\"Revenue_m\"] = plot_df.Revenue / 1e6
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.boxplot(
    data=plot_df, x=\"promo_bucket\", y=\"Revenue_m\",
    palette=\"RdBu\", showfliers=False,
    hue=\"promo_bucket\", legend=False,
    ax=ax,
)
ax.set_xlabel(\"% đơn có promo trong ngày\")
ax.set_ylabel(\"Revenue (triệu VND)\")
ax.set_title(\"Ngày promo dày → doanh thu thấp hơn, không cao hơn\")
save_report(\"promo_share_vs_revenue.png\")
"""
))

CELLS.append(code(
    """# Promo depth effect: weekly avg discount per line vs revenue
weekly_promo = merged_promo.resample(\"W\").agg({\n    \"Revenue\": \"sum\",\n    \"total_discount\": \"sum\",\n    \"total_lines\": \"sum\",\n    \"promo_lines\": \"sum\",\n})
weekly_promo[\"discount_per_line\"] = weekly_promo.total_discount / weekly_promo.total_lines.replace(0, np.nan)
weekly_promo[\"promo_share\"] = weekly_promo.promo_lines / weekly_promo.total_lines.replace(0, np.nan)
weekly_promo = weekly_promo.dropna()

fig, ax = plt.subplots(figsize=(9, 4.5))
sc = ax.scatter(\n    weekly_promo.promo_share * 100,\n    weekly_promo.Revenue / 1e9,\n    c=weekly_promo.discount_per_line,\n    cmap=\"viridis\", s=14, alpha=0.7,\n)
ax.set_xlabel(\"% đơn có promo (weekly)\")
ax.set_ylabel(\"Revenue (tỷ VND, weekly)\")
ax.set_title(\"Tuần promo dày = tuần doanh thu thấp (màu = giảm giá/line)\")
plt.colorbar(sc, label=\"Discount TB / line\")
save_eda(\"10_promo_weekly_discount.png\")
"""
))


# ---------------------------------------------------------------------------
# 8. Returns / refunds
# ---------------------------------------------------------------------------

CELLS.append(md("## 8. Trả hàng & hoàn tiền"))

CELLS.append(code(
    """ret_full = returns.merge(
    products[[\"product_id\", \"category\", \"segment\"]], on=\"product_id\", how=\"left\"
)
sold_by_cat = oi_enriched.groupby(\"category\").quantity.sum()
refund_by_cat = ret_full.groupby(\"category\").refund_amount.sum() / 1e9
ret_qty_cat = ret_full.groupby(\"category\").return_quantity.sum()
cat_tab = pd.DataFrame({
    \"sold_qty\": sold_by_cat,
    \"return_qty\": ret_qty_cat,
    \"refund_bn\": refund_by_cat,
})
cat_tab[\"return_rate\"] = cat_tab.return_qty / cat_tab.sold_qty * 100
cat_tab = cat_tab.sort_values(\"return_rate\", ascending=False)
display(cat_tab)
log(\"return_rate_by_category\", cat_tab.reset_index())
"""
))

CELLS.append(code(
    """# Return-reason breakdown
reason = ret_full.return_reason.value_counts(normalize=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
cat_tab.return_rate.plot(kind=\"bar\", ax=axes[0], color=\"#e53935\")
axes[0].set_title(\"Return-rate theo category (%)\")
axes[0].set_ylabel(\"%\")
axes[0].tick_params(axis=\"x\", rotation=20)
reason.plot(kind=\"bar\", ax=axes[1], color=\"#f57c00\")
axes[1].set_title(\"Phân phối lý do trả hàng (%)\")
axes[1].set_ylabel(\"%\")
axes[1].tick_params(axis=\"x\", rotation=25)
save_eda(\"11_returns_category_reason.png\")
log(\"return_reason_pct\", reason)
"""
))

CELLS.append(code(
    """# Monthly refund vs revenue — did refunds grow?
returns[\"month\"] = returns.return_date.values.astype(\"datetime64[M]\")
refund_m = returns.groupby(\"month\").refund_amount.sum() / 1e9
rev_m = sales.set_index(\"Date\").Revenue.resample(\"MS\").sum() / 1e9
joined = pd.concat([rev_m.rename(\"revenue\"), refund_m.rename(\"refund\")], axis=1).fillna(0)
joined[\"refund_ratio_%\"] = joined.refund / joined.revenue * 100

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(joined.index, joined[\"refund_ratio_%\"], color=\"#c62828\", linewidth=1.4)
ax.set_title(\"Tỷ lệ refund / doanh thu theo tháng (%)\")
ax.set_ylabel(\"%\")
save_eda(\"12_refund_ratio_monthly.png\")
log(\"refund_ratio_mean\", float(joined[\"refund_ratio_%\"].replace([np.inf, -np.inf], np.nan).mean()))
"""
))


# ---------------------------------------------------------------------------
# 9. Inventory
# ---------------------------------------------------------------------------

CELLS.append(md("## 9. Tồn kho — có nghẽn cung không?"))

CELLS.append(code(
    """inv = inventory.copy()
inv[\"ym\"] = pd.to_datetime(inv.snapshot_date).dt.to_period(\"M\").dt.to_timestamp()
inv_month = inv.groupby(\"ym\").agg(
    n_sku=(\"product_id\", \"count\"),
    stockout_rate=(\"stockout_flag\", \"mean\"),
    overstock_rate=(\"overstock_flag\", \"mean\"),
    reorder_rate=(\"reorder_flag\", \"mean\"),
    mean_fill_rate=(\"fill_rate\", \"mean\"),
    mean_days_of_supply=(\"days_of_supply\", \"mean\"),
)
display(inv_month.tail(12))
log(\"inv_last12m\", inv_month.tail(12).reset_index())
"""
))

CELLS.append(code(
    """fig, ax1 = plt.subplots(figsize=(13, 4.2))
ax1.plot(inv_month.index, inv_month.stockout_rate * 100, color=\"#c62828\", label=\"Stockout rate\")
ax1.plot(inv_month.index, inv_month.overstock_rate * 100, color=\"#1565c0\", label=\"Overstock rate\")
ax1.set_ylabel(\"% SKU (stockout / overstock)\")
ax2 = ax1.twinx()
ax2.plot(inv_month.index, inv_month.mean_fill_rate * 100, color=\"#2e7d32\", linewidth=1.8, label=\"Fill rate (RHS)\")
ax2.set_ylabel(\"Fill rate (%)\")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc=\"lower left\")
ax1.set_title(\"Stockout và overstock cùng tăng — paradox phân bổ tồn kho\")
save_report(\"inventory_health_monthly.png\")
"""
))

CELLS.append(code(
    """# Does monthly stockout_rate correlate with total monthly revenue?
rev_month = sales.set_index(\"Date\").Revenue.resample(\"MS\").sum()
join = inv_month.join(rev_month.rename(\"revenue\"), how=\"inner\")
corr = join[[\"revenue\", \"stockout_rate\", \"overstock_rate\", \"mean_fill_rate\", \"mean_days_of_supply\"]].corr(\"spearman\").iloc[0]
display(corr.to_frame(\"spearman vs revenue\"))
log(\"inv_revenue_spearman\", corr)
"""
))

CELLS.append(code(
    """# Top category stockout pattern
inv_cat = inv.groupby([\"ym\", \"category\"]).stockout_flag.mean().unstack() * 100
plt.figure(figsize=(11, 4.5))
sns.heatmap(inv_cat.T, cmap=\"Reds\", cbar_kws={\"label\": \"% SKU stockout\"})
plt.title(\"Stockout % theo category × tháng\")
plt.xlabel(\"\"); plt.ylabel(\"\")
save_eda(\"13_stockout_by_category_month.png\")
"""
))


# ---------------------------------------------------------------------------
# 10. Web traffic vs revenue
# ---------------------------------------------------------------------------

CELLS.append(md("## 10. Web traffic & mối quan hệ với Revenue"))

CELLS.append(code(
    """web_day = web.groupby(\"date\").agg(
    sessions=(\"sessions\", \"sum\"),
    unique_visitors=(\"unique_visitors\", \"sum\"),
    page_views=(\"page_views\", \"sum\"),
    bounce_rate=(\"bounce_rate\", \"mean\"),
    avg_session_duration_sec=(\"avg_session_duration_sec\", \"mean\"),
)
web_day.index = pd.to_datetime(web_day.index)
wm = sales[[\"Date\", \"Revenue\"]].set_index(\"Date\").join(web_day, how=\"inner\")

corr_cols = [\"Revenue\", \"sessions\", \"unique_visitors\", \"page_views\", \"bounce_rate\", \"avg_session_duration_sec\"]
sp = wm[corr_cols].corr(\"spearman\").Revenue.drop(\"Revenue\")
display(sp.to_frame(\"Spearman với Revenue\"))
log(\"web_revenue_spearman\", sp)
"""
))

CELLS.append(code(
    """# Per traffic_source correlation (explodes average)
web_src = web.pivot_table(index=\"date\", columns=\"traffic_source\", values=\"sessions\", aggfunc=\"sum\").fillna(0)
web_src.index = pd.to_datetime(web_src.index)
ws = sales[[\"Date\", \"Revenue\"]].set_index(\"Date\").join(web_src, how=\"inner\")
per_src = ws.corr(\"spearman\").Revenue.drop(\"Revenue\").sort_values(ascending=False)
display(per_src.to_frame(\"Spearman Revenue vs sessions / kênh\"))
log(\"web_source_spearman\", per_src)
"""
))

CELLS.append(code(
    """fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(wm.sessions, wm.Revenue / 1e6, alpha=0.3, s=6, color=\"#1565c0\")
ax.set_xlabel(\"Sessions / ngày\")
ax.set_ylabel(\"Revenue (triệu VND)\")
ax.set_title(f\"Web sessions vs Revenue (ρ_Spearman = {sp.loc['sessions']:.2f})\")
save_report(\"webtraffic_vs_revenue.png\")
"""
))

CELLS.append(code(
    """# Lead-lag: does traffic lead revenue?
best_lag = {}
for col in [\"sessions\", \"unique_visitors\", \"page_views\"]:
    results = {}
    for k in range(-7, 8):
        x = wm[col].shift(k)
        y = wm.Revenue
        mask = x.notna() & y.notna()
        results[k] = stats.spearmanr(x[mask], y[mask]).correlation
    best_lag[col] = results
lag_df = pd.DataFrame(best_lag)
display(lag_df)
log(\"web_leadlag\", lag_df)

fig, ax = plt.subplots(figsize=(9, 4))
for col in lag_df.columns:
    ax.plot(lag_df.index, lag_df[col], marker=\"o\", label=col)
ax.axvline(0, color=\"grey\", linestyle=\"--\")
ax.set_xlabel(\"Lag (ngày)  — âm = traffic LEAD revenue\")
ax.set_ylabel(\"Spearman ρ\")
ax.set_title(\"Lead-lag traffic ↔ revenue (chọn k = argmax ρ)\")
ax.legend()
save_eda(\"14_web_leadlag_correlation.png\")
"""
))


# ---------------------------------------------------------------------------
# 11. Shipments & reviews
# ---------------------------------------------------------------------------

CELLS.append(md("## 11. Vận chuyển & đánh giá"))

CELLS.append(code(
    """ship = shipments.copy()
ship[\"delivery_days\"] = (ship.delivery_date - ship.ship_date).dt.days
ship_stats = ship.delivery_days.describe()
display(ship_stats.to_frame())
log(\"delivery_days_stats\", ship_stats)

rev_ship = ship.merge(reviews[[\"order_id\", \"rating\"]], on=\"order_id\", how=\"inner\")
rev_ship[\"bucket\"] = pd.cut(rev_ship.delivery_days, bins=[-1, 2, 4, 6, 9, 30],
                              labels=[\"0–2\", \"3–4\", \"5–6\", \"7–9\", \"10+\"])
rating_by_bucket = rev_ship.groupby(\"bucket\", observed=True).rating.agg([\"count\", \"mean\", lambda r: (r <= 2).mean() * 100])
rating_by_bucket.columns = [\"n\", \"mean_rating\", \"pct_low_rating\"]
display(rating_by_bucket)
log(\"rating_by_delivery_bucket\", rating_by_bucket.reset_index())
"""
))

CELLS.append(code(
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(ship.delivery_days.dropna(), bins=30, ax=axes[0], color=\"#1565c0\")
axes[0].set_title(\"Phân phối thời gian giao hàng\")
axes[0].set_xlabel(\"Ngày\")

rating_by_bucket.mean_rating.plot(kind=\"bar\", ax=axes[1], color=\"#2e7d32\")
axes[1].set_ylim(3.5, 4.1)
axes[1].set_title(\"Rating TB theo bucket delivery_days\")
axes[1].tick_params(axis=\"x\", rotation=0)
save_report(\"delivery_vs_rating.png\")
"""
))

CELLS.append(code(
    """# Review rating distribution, overall mean
rating_dist = reviews.rating.value_counts(normalize=True).sort_index() * 100
print(\"Overall mean rating:\", reviews.rating.mean())
display(rating_dist.to_frame(\"%\"))
log(\"rating_distribution_pct\", rating_dist)

# ANOVA: is rating different across delivery buckets? (truly validate Insight 8)
groups = [rev_ship.loc[rev_ship.bucket == b, \"rating\"].dropna() for b in rating_by_bucket.index]
f_stat, p_val = stats.f_oneway(*groups)
print(f\"ANOVA rating ~ bucket: F={f_stat:.2f}  p={p_val:.2e}\")
log(\"rating_anova\", {\"f_stat\": float(f_stat), \"p_value\": float(p_val)})
"""
))


# ---------------------------------------------------------------------------
# 12. Region
# ---------------------------------------------------------------------------

CELLS.append(md("## 12. Địa lý: doanh thu theo region"))

CELLS.append(code(
    """oi_geo = oi_enriched.merge(orders[[\"order_id\", \"zip\"]], on=\"order_id\", how=\"left\") \\
                      .merge(geography[[\"zip\", \"region\"]], on=\"zip\", how=\"left\")
reg_rev = oi_geo.groupby(\"region\").line_rev.sum().sort_values(ascending=False) / 1e9
display(reg_rev.to_frame(\"Revenue (tỷ VND)\"))
log(\"region_revenue_bn\", reg_rev)
"""
))

CELLS.append(code(
    """# Region HHI
rs = reg_rev / reg_rev.sum()
hhi_reg = (rs ** 2).sum()
print(f\"Region HHI = {hhi_reg:.3f}\")
log(\"region_hhi\", float(hhi_reg))
"""
))


# ---------------------------------------------------------------------------
# 13. Forecast readiness
# ---------------------------------------------------------------------------

CELLS.append(md(
    """## 13. Forecast readiness & exogenous signal ranking

Tổng hợp mối quan hệ Spearman của các *tín hiệu ngoại sinh* với `Revenue` hàng ngày (2012–2022). Mục đích: xếp hạng ưu tiên feature cho mô hình.
"""
))

CELLS.append(code(
    """# Build daily panel with all exogenous features
daily = sales.set_index(\"Date\")[[\"Revenue\", \"COGS\"]].copy()
daily = daily.join(daily_orders.rename(\"n_orders\"), how=\"left\")
# Daily promo share from oi_with_date
daily = daily.join(daily_promo[[\"promo_share\", \"total_discount\", \"promo_lines\"]], how=\"left\")
daily = daily.join(web_day[[\"sessions\", \"unique_visitors\", \"page_views\", \"bounce_rate\"]], how=\"left\")
daily = daily.join(
    returns.set_index(\"return_date\").refund_amount.resample(\"D\").sum().rename(\"refund\"), how=\"left\"
)
# Monthly features forward-filled to daily
inv_daily = inv_month.reindex(pd.date_range(inv_month.index.min(), daily.index.max()), method=\"ffill\")
daily = daily.join(inv_daily[[\"stockout_rate\", \"overstock_rate\", \"mean_fill_rate\"]], how=\"left\")

daily = daily.fillna({c: 0 for c in [\"n_orders\", \"promo_share\", \"total_discount\", \"promo_lines\", \"refund\"]})
daily[\"margin\"] = 1 - daily.COGS / daily.Revenue

display(daily.tail())
log(\"panel_shape\", list(daily.shape))
"""
))

CELLS.append(code(
    """sp_all = daily.corr(\"spearman\").Revenue.drop(\"Revenue\").sort_values(key=abs, ascending=False)
display(sp_all.to_frame(\"Spearman ρ với Revenue\"))
log(\"panel_spearman_revenue\", sp_all)

fig, ax = plt.subplots(figsize=(9, 5))
sp_all.plot(kind=\"barh\", color=[\"#c62828\" if v < 0 else \"#1565c0\" for v in sp_all.values], ax=ax)
ax.set_title(\"Exogenous features — Spearman ρ với Revenue\")
ax.axvline(0, color=\"grey\")
save_report(\"exogenous_signal_ranking.png\")
"""
))

CELLS.append(code(
    """# Stationarity hint: rolling mean & std of Revenue
roll = daily.Revenue.rolling(180)
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(daily.index, roll.mean()/1e6, color=\"#1565c0\", label=\"Rolling mean (180d)\")
ax.plot(daily.index, roll.std()/1e6, color=\"#c62828\", label=\"Rolling std (180d)\")
ax.set_ylabel(\"Triệu VND\")
ax.legend()
ax.set_title(\"Rolling mean & std — Revenue không dừng (mean giảm, std khá ổn định)\")
save_eda(\"15_revenue_stationarity.png\")
"""
))

CELLS.append(code(
    """# Test horizon chart — for report
test_start = pd.Timestamp(\"2023-01-01\")
test_end = pd.Timestamp(\"2024-07-01\")
quarterly = sales.groupby(pd.Grouper(key=\"Date\", freq=\"QS\")).Revenue.sum() / 1e9

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(quarterly.index, quarterly.values, width=70, color=\"#37474f\")
ax.axvspan(test_start, test_end, color=\"#ef6c00\", alpha=0.3, label=f\"Test horizon ({(test_end-test_start).days} ngày)\")
ax.set_title(\"Doanh thu quý + horizon dự báo 2023-01 → 2024-07\")
ax.set_ylabel(\"Revenue (tỷ VND)\")
ax.legend(loc=\"upper right\")
save_report(\"quarterly_revenue_with_test_horizon.png\")
"""
))


# ---------------------------------------------------------------------------
# 14. Export metrics
# ---------------------------------------------------------------------------

CELLS.append(md("## 14. Lưu metrics"))

CELLS.append(code(
    """metrics_path = OUT / \"metrics.json\"
with open(metrics_path, \"w\") as f:
    json.dump(METRICS, f, indent=2, default=str, ensure_ascii=False)
print(\"saved metrics to:\", metrics_path)
print(\"# keys:\", len(METRICS))
print(sorted(METRICS.keys()))
"""
))


NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.13",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)
print("Wrote", NB_PATH, "cells:", len(CELLS))
