"""Big FE overhaul v4 — all aggregation modules.

Every aggregation produces a per-Date frame. The final
`build_exog_v4(all_dates)` joins everything to a dense daily index.

No values from `sales_test.csv` or `sample_submission.csv` are read.
All per-day numbers reflect the train period only.

Leakage policy: any feature whose value on day `t` semantically equals
the target of day `t` (e.g. `items_cogs_total_value` which equals COGS
by construction) is placed in `LEAKY_LEVEL_COLS`. The model runner
applies day-of-year mean imputation to these columns consistently
across train and inference so the daily distribution is stable.

Features cover categories A, B, C, D, E, F, H, I, J from the plan.
VN calendar (G) lives in `legacy_components/calendar_vn.py`.
Aux margin model (K) lives in `legacy_components/final_model_v4.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
DATA = ROOT.parent / "data"


# Known categoricals (determined from EDA):
TOP_CATEGORIES = ["Streetwear", "Outdoor", "Casual", "GenZ"]
TOP_SEGMENTS = [
    "Activewear", "Everyday", "Performance", "Balanced",
    "Standard", "Premium", "All-weather", "Trendy",
]
PAY_METHODS = ["credit_card", "paypal", "cod", "apple_pay", "bank_transfer"]
DEVICES = ["mobile", "desktop", "tablet"]
SOURCES = ["organic_search", "paid_search", "social_media", "email_campaign", "referral", "direct"]
REGIONS = ["East", "Central", "West"]


# Columns whose raw value on day `t` is essentially the target of day `t`
# (so must be DoY-mean-imputed consistently, not used directly).
LEAKY_LEVEL_COLS_V4 = {
    # direct level leaks (perfect or near-perfect correlation with target)
    "items_cogs_total_value",      # = COGS exactly (sum quantity*cogs)
    "items_gross_value",            # = Revenue (legacy from v1)
    "pay_total_value",              # ≈ Revenue * 1.06
    "items_discount_total",
    "pay_mean_value",
    # near-leak counts (|corr| > 0.94) that scale linearly with revenue:
    # replaced consistently with DoY mean to stabilise the distribution.
    "orders_count",
    "orders_unique_customers",
    "orders_unique_zips",
    "repeat_orders",
    "first_time_orders",
    "items_total_qty_v4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_daily(df: pd.DataFrame, date_col: str, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame({"Date": all_dates}).merge(
        df.rename(columns={date_col: "Date"}), on="Date", how="left"
    )
    return out


# ---------------------------------------------------------------------------
# A. Category / segment / size / color breakdown
# ---------------------------------------------------------------------------


def _agg_items_by_mix(orders: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Daily per-category/segment/size breakdown from order_items + products."""
    df = items.merge(products[["product_id", "category", "segment", "size", "cogs", "price"]],
                     on="product_id", how="left")
    df = df.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    df["gross"] = df["quantity"] * df["unit_price"]
    df["cogs_val"] = df["quantity"] * df["cogs"]
    df["is_L_XL"] = df["size"].isin(["L", "XL"]).astype(int)
    df["is_S_M"] = df["size"].isin(["S", "M"]).astype(int)

    day = df.groupby("order_date")
    out = day.agg(
        items_total_qty_v4=("quantity", "sum"),
        items_gross_value=("gross", "sum"),
        items_cogs_total_value=("cogs_val", "sum"),
        items_discount_total=("discount_amount", "sum"),
        items_avg_unit_price=("unit_price", "mean"),
        items_unique_categories=("category", "nunique"),
        items_unique_segments=("segment", "nunique"),
        items_unique_products=("product_id", "nunique"),
        items_size_LXL_share=("is_L_XL", "mean"),
        items_size_SM_share=("is_S_M", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})

    # Per-category quantity share
    cat_daily = (
        df.groupby(["order_date", "category"])["quantity"].sum().unstack(fill_value=0)
    )
    cat_totals = cat_daily.sum(axis=1).replace(0, np.nan)
    for cat in TOP_CATEGORIES:
        col = f"items_qty_share_cat_{cat.lower()}"
        out[col] = out["Date"].map((cat_daily.get(cat, 0) / cat_totals).to_dict()).fillna(0)

    # Per-segment share
    seg_daily = (
        df.groupby(["order_date", "segment"])["quantity"].sum().unstack(fill_value=0)
    )
    seg_totals = seg_daily.sum(axis=1).replace(0, np.nan)
    for seg in TOP_SEGMENTS:
        col = f"items_qty_share_seg_{seg.lower().replace('-', '_')}"
        out[col] = out["Date"].map((seg_daily.get(seg, 0) / seg_totals).to_dict()).fillna(0)

    return out


# ---------------------------------------------------------------------------
# B. Basket / price ratios — derived inside build_exog_v4 from A outputs.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C. RFM / customer lifecycle
# ---------------------------------------------------------------------------


def _agg_customer_lifecycle(orders: pd.DataFrame, customers: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute RFM-style daily features.

    active_customers_{W} uses a sliding window up to and including day t
    (no future info). Done vectorized per customer.
    """
    cust = customers[["customer_id", "signup_date"]].copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    ord2 = orders.merge(cust, on="customer_id", how="left")
    ord2 = ord2.dropna(subset=["order_date"]).copy()
    ord2["order_date"] = pd.to_datetime(ord2["order_date"])
    ord2 = ord2.sort_values(["customer_id", "order_date"])
    ord2["order_rank"] = ord2.groupby("customer_id").cumcount() + 1
    ord2["cust_age_days"] = (ord2["order_date"] - ord2["signup_date"]).dt.days.clip(lower=0)

    # Per-day aggregates from orders
    daily = (
        ord2.groupby("order_date")
        .agg(
            first_time_orders=("order_rank", lambda s: (s == 1).sum()),
            repeat_orders=("order_rank", lambda s: (s > 1).sum()),
            customer_age_mean=("cust_age_days", "mean"),
            signup_to_first_leadtime_mean=(
                "cust_age_days",
                lambda s: s[ord2.loc[s.index, "order_rank"] == 1].mean(),
            ),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    daily["first_time_buyer_rate"] = daily["first_time_orders"] / (
        daily["first_time_orders"] + daily["repeat_orders"]
    ).replace(0, np.nan)

    # Sliding active customers: count unique customer per window.
    # Efficient approach: per-day set of customer ids, then rolling sets
    # are too memory-heavy. Approximation: count unique customer_id
    # whose order_date is within [t - W + 1, t].
    # We build it via cumulative "last-seen" approach.
    cust_days = ord2[["customer_id", "order_date"]].drop_duplicates()
    cust_days = cust_days.sort_values("order_date").reset_index(drop=True)

    all_days = pd.date_range(cust_days["order_date"].min(), all_dates.max(), freq="D")
    day_index = {d: i for i, d in enumerate(all_days)}

    def _rolling_active(window: int) -> pd.Series:
        # For each customer, mark all days in [first_seen, last_seen + window] as active.
        # Simpler: count of unique customers with at least one order in [t-W+1, t].
        # Use 1D sweep over sorted events.
        events = []
        for cid, grp in cust_days.groupby("customer_id"):
            dates = grp["order_date"].to_numpy()
            for d in dates:
                start = d
                end = d + pd.Timedelta(days=window - 1)
                events.append((start, 1, cid))
                events.append((end + pd.Timedelta(days=1), -1, cid))
        # Fallback: straight count unique customer per day window via groupby expansions.
        # For memory/perf safety, use pandas rolling on per-day "sets" approximated by
        # sum of unique first-appearances in window. We use union approximation:
        raise NotImplementedError

    # Simpler & correct: for each day, count customers whose most recent order
    # within the previous W days is present. Use a per-customer rolling bitmap
    # via pandas reindex.
    def _active_per_window(window_days: int) -> pd.Series:
        tmp = cust_days.copy()
        tmp["one"] = 1
        pivot = tmp.pivot_table(
            index="order_date", columns="customer_id", values="one", aggfunc="max", fill_value=0
        )
        pivot = pivot.reindex(all_days, fill_value=0)
        rolled = pivot.rolling(window=window_days, min_periods=1).max()
        # Number of customers "seen" in window = row sum of rolled bitmap
        return rolled.sum(axis=1)

    # Memory check: customers ~ 100k → 2.5M days * 100k = too big. Use approximation:
    # active_customers_W ≈ unique customer_id count in window, computed via
    # exploding per-order rows into a list of (date, cid) and grouping.

    def _active_fast(window_days: int) -> pd.Series:
        # Explode each order into a date it contributes to: same date only.
        # Count unique customers per rolling window using a hash-based streaming method.
        ev = cust_days.copy()
        ev = ev.sort_values("order_date").reset_index(drop=True)
        ev["d"] = ev["order_date"]
        res = {}
        # bucket by date
        per_day = ev.groupby("d")["customer_id"].apply(set)
        day_list = sorted(per_day.index)
        cursor_start = 0
        window = {}  # cid -> last_day_index
        active = set()
        result = pd.Series(0, index=all_days, dtype=np.int64)
        day_to_order = {d: i for i, d in enumerate(day_list)}
        for end_day in all_days:
            # Add all days <= end_day not yet added
            while cursor_start < len(day_list) and day_list[cursor_start] <= end_day:
                d = day_list[cursor_start]
                for cid in per_day[d]:
                    window[cid] = d
                    active.add(cid)
                cursor_start += 1
            # Drop anything older than window_days
            threshold = end_day - pd.Timedelta(days=window_days - 1)
            stale = [cid for cid, last in window.items() if last < threshold]
            for cid in stale:
                window.pop(cid, None)
                active.discard(cid)
            result.loc[end_day] = len(active)
        return result

    for W in (28, 90, 365):
        s = _active_fast(W)
        daily[f"active_customers_{W}d"] = daily["Date"].map(s.to_dict())

    return daily


# ---------------------------------------------------------------------------
# D. Geography share
# ---------------------------------------------------------------------------


def _agg_geography(orders: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    geo = geography[["zip", "region", "city"]].drop_duplicates(subset=["zip"])
    df = orders.merge(geo, on="zip", how="left")
    top_cities = df["city"].value_counts().head(5).index.tolist()
    df["is_top_city"] = df["city"].isin(top_cities).astype(int)

    g = df.groupby("order_date")
    out = g.agg(
        orders_unique_cities=("city", "nunique"),
        orders_unique_zips=("zip", "nunique"),
        orders_top5_city_share=("is_top_city", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})

    # Per-region share
    reg_daily = df.groupby(["order_date", "region"]).size().unstack(fill_value=0)
    totals = reg_daily.sum(axis=1).replace(0, np.nan)
    for r in REGIONS:
        col = f"orders_region_{r.lower()}_share"
        out[col] = out["Date"].map((reg_daily.get(r, 0) / totals).to_dict()).fillna(0)

    return out


# ---------------------------------------------------------------------------
# E. Payment mix
# ---------------------------------------------------------------------------


def _agg_payments_v4(payments: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    pay = payments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    # Overall aggregates
    g = pay.groupby("order_date").agg(
        pay_total_value=("payment_value", "sum"),
        pay_mean_value=("payment_value", "mean"),
        pay_mean_installments=("installments", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})

    # Installment buckets
    inst = pay.copy()
    inst["inst_gt3"] = (inst["installments"] > 3).astype(int)
    inst_daily = inst.groupby("order_date")["inst_gt3"].mean().rename("pay_installments_gt3_share")
    g = g.merge(inst_daily.reset_index().rename(columns={"order_date": "Date"}), on="Date", how="left")

    # Payment method share (quantity-based)
    meth_daily = pay.groupby(["order_date", "payment_method"]).size().unstack(fill_value=0)
    totals = meth_daily.sum(axis=1).replace(0, np.nan)
    for m in PAY_METHODS:
        col = f"pay_share_{m}"
        g[col] = g["Date"].map((meth_daily.get(m, 0) / totals).to_dict()).fillna(0)

    return g


# ---------------------------------------------------------------------------
# F. Returns / reviews joined to order_date
# ---------------------------------------------------------------------------


def _agg_returns_on_order(returns: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = returns.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    df = df.dropna(subset=["order_date"])
    g = df.groupby("order_date").agg(
        returns_by_order_count=("return_id", "count"),
        returns_by_order_qty=("return_quantity", "sum"),
        returns_by_order_refund=("refund_amount", "sum"),
    ).reset_index().rename(columns={"order_date": "Date"})
    return g


def _agg_reviews_on_order(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = reviews.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    df = df.dropna(subset=["order_date"])
    df["is_bad"] = (df["rating"] <= 2).astype(int)
    df["is_good"] = (df["rating"] >= 4).astype(int)
    g = df.groupby("order_date").agg(
        reviews_by_order_count=("review_id", "count"),
        reviews_by_order_rating=("rating", "mean"),
        reviews_by_order_bad_rate=("is_bad", "mean"),
        reviews_by_order_good_rate=("is_good", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})
    return g


# ---------------------------------------------------------------------------
# H. (wrap) core orders agg — extended version
# ---------------------------------------------------------------------------


def _agg_orders_v4(orders: pd.DataFrame) -> pd.DataFrame:
    g = orders.groupby("order_date").agg(
        orders_count=("order_id", "count"),
        orders_unique_customers=("customer_id", "nunique"),
    ).reset_index().rename(columns={"order_date": "Date"})

    for d in DEVICES:
        mask_df = orders.copy()
        mask_df["_m"] = (mask_df["device_type"] == d).astype(int)
        share = mask_df.groupby("order_date")["_m"].mean().rename(f"orders_device_{d}_share")
        g = g.merge(share.reset_index().rename(columns={"order_date": "Date"}), on="Date", how="left")

    for s in SOURCES:
        mask_df = orders.copy()
        mask_df["_m"] = (mask_df["order_source"] == s).astype(int)
        share = mask_df.groupby("order_date")["_m"].mean().rename(f"orders_source_{s}_share")
        g = g.merge(share.reset_index().rename(columns={"order_date": "Date"}), on="Date", how="left")

    orders_status = orders.copy()
    orders_status["is_delivered"] = (orders_status["order_status"] == "delivered").astype(int)
    orders_status["is_cancelled"] = (orders_status["order_status"] == "cancelled").astype(int)
    orders_status["is_returned_status"] = (orders_status["order_status"] == "returned").astype(int)
    st = orders_status.groupby("order_date").agg(
        orders_delivered_share=("is_delivered", "mean"),
        orders_cancelled_share=("is_cancelled", "mean"),
        orders_returned_status_share=("is_returned_status", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})
    g = g.merge(st, on="Date", how="left")
    return g


# ---------------------------------------------------------------------------
# J. Promotion depth
# ---------------------------------------------------------------------------


def _agg_promotions_v4(promotions: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    promos = promotions.copy()
    promos["start_date"] = pd.to_datetime(promos["start_date"])
    promos["end_date"] = pd.to_datetime(promos["end_date"])

    out = pd.DataFrame({"Date": all_dates})
    out["promo_active_count"] = 0
    out["promo_max_discount"] = 0.0
    out["promo_mean_discount"] = 0.0
    out["promo_pct_type_count"] = 0
    out["promo_fixed_type_count"] = 0
    out["promo_category_specific_count"] = 0
    out["promo_stackable_count"] = 0

    # Vectorize by expanding events
    # For each promo, increment counters on days in [start, end]
    for _, r in promos.iterrows():
        mask = out["Date"].between(r["start_date"], r["end_date"])
        out.loc[mask, "promo_active_count"] += 1
        disc = float(r.get("discount_value", 0) or 0)
        out.loc[mask, "promo_max_discount"] = np.maximum(out.loc[mask, "promo_max_discount"], disc)
        out.loc[mask, "promo_mean_discount"] += disc
        if r.get("promo_type") == "percentage":
            out.loc[mask, "promo_pct_type_count"] += 1
        elif r.get("promo_type") == "fixed":
            out.loc[mask, "promo_fixed_type_count"] += 1
        if pd.notna(r.get("applicable_category", None)):
            out.loc[mask, "promo_category_specific_count"] += 1
        if float(r.get("stackable_flag", 0) or 0) > 0:
            out.loc[mask, "promo_stackable_count"] += 1

    active_nz = out["promo_active_count"].replace(0, np.nan)
    out["promo_mean_discount"] = out["promo_mean_discount"] / active_nz
    out["promo_mean_discount"] = out["promo_mean_discount"].fillna(0)
    out["promo_pct_type_share"] = out["promo_pct_type_count"] / active_nz
    out["promo_fixed_type_share"] = out["promo_fixed_type_count"] / active_nz
    out["promo_pct_type_share"] = out["promo_pct_type_share"].fillna(0)
    out["promo_fixed_type_share"] = out["promo_fixed_type_share"].fillna(0)
    out["promo_active"] = (out["promo_active_count"] > 0).astype(int)

    # Days since last promo start / to next promo start
    starts = sorted(promos["start_date"].dropna().unique())
    starts_ts = pd.DatetimeIndex(starts)
    all_idx = pd.DatetimeIndex(all_dates)

    def _days_since_prev(events: pd.DatetimeIndex) -> np.ndarray:
        ev_arr = events.values.astype("datetime64[D]").astype(np.int64)
        d_arr = all_idx.values.astype("datetime64[D]").astype(np.int64)
        ev_sorted = np.sort(ev_arr)
        idx = np.searchsorted(ev_sorted, d_arr, side="right") - 1
        out_arr = np.full(len(d_arr), 365, dtype=float)
        valid = idx >= 0
        out_arr[valid] = np.clip(d_arr[valid] - ev_sorted[idx[valid]], 0, 365)
        return out_arr

    def _days_to_next(events: pd.DatetimeIndex) -> np.ndarray:
        ev_arr = events.values.astype("datetime64[D]").astype(np.int64)
        d_arr = all_idx.values.astype("datetime64[D]").astype(np.int64)
        ev_sorted = np.sort(ev_arr)
        idx = np.searchsorted(ev_sorted, d_arr, side="left")
        out_arr = np.full(len(d_arr), 365, dtype=float)
        valid = idx < len(ev_sorted)
        out_arr[valid] = np.clip(ev_sorted[idx[valid]] - d_arr[valid], 0, 365)
        return out_arr

    out["days_since_last_promo_start"] = _days_since_prev(starts_ts)
    out["days_to_next_promo_start"] = _days_to_next(starts_ts)

    return out


# ---------------------------------------------------------------------------
# I. Inventory dynamics (extends v1 agg)
# ---------------------------------------------------------------------------


def _agg_inventory_v4(inventory: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    inv = inventory.copy()
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    monthly = inv.groupby("snapshot_date").agg(
        inv_stockout_rate=("stockout_flag", "mean"),
        inv_overstock_rate=("overstock_flag", "mean"),
        inv_reorder_rate=("reorder_flag", "mean"),
        inv_fill_rate=("fill_rate", "mean"),
        inv_sell_through=("sell_through_rate", "mean"),
        inv_days_of_supply=("days_of_supply", "mean"),
    ).reset_index().rename(columns={"snapshot_date": "Date"})

    out = pd.DataFrame({"Date": all_dates}).merge(monthly, on="Date", how="left")
    cols = [c for c in out.columns if c != "Date"]
    out[cols] = out[cols].ffill()

    # Derived: reorder intensity and diffs
    out["inv_reorder_intensity"] = out["inv_reorder_rate"] * out["inv_stockout_rate"]
    out["inv_stockout_lag_30"] = out["inv_stockout_rate"].shift(30)
    out["inv_stockout_lag_90"] = out["inv_stockout_rate"].shift(90)
    out["inv_days_of_supply_diff_30"] = out["inv_days_of_supply"].diff(30)
    out["inv_fill_rate_diff_30"] = out["inv_fill_rate"].diff(30)
    return out


# ---------------------------------------------------------------------------
# Small reused v1-ish aggregations
# ---------------------------------------------------------------------------


def _agg_shipments_v4(shipments: pd.DataFrame) -> pd.DataFrame:
    sh = shipments.copy()
    sh["ship_date"] = pd.to_datetime(sh["ship_date"])
    sh["delivery_date"] = pd.to_datetime(sh["delivery_date"])
    sh["leadtime"] = (sh["delivery_date"] - sh["ship_date"]).dt.days
    g = sh.groupby("ship_date").agg(
        ship_count=("order_id", "count"),
        ship_fee_total=("shipping_fee", "sum"),
        ship_fee_mean=("shipping_fee", "mean"),
        ship_leadtime_mean=("leadtime", "mean"),
        ship_leadtime_std=("leadtime", "std"),
    ).reset_index().rename(columns={"ship_date": "Date"})
    return g


def _agg_web_v4(web: pd.DataFrame) -> pd.DataFrame:
    df = web.copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby("date").agg(
        web_sessions=("sessions", "sum"),
        web_unique_visitors=("unique_visitors", "sum"),
        web_page_views=("page_views", "sum"),
        web_bounce_rate=("bounce_rate", "mean"),
        web_avg_session=("avg_session_duration_sec", "mean"),
    ).reset_index().rename(columns={"date": "Date"})
    g["web_pv_per_session"] = g["web_page_views"] / g["web_sessions"].replace(0, np.nan)
    g["web_pv_per_session"] = g["web_pv_per_session"].fillna(0)
    return g


def _agg_customers_new(customers: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    cust = customers.copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    daily = cust.groupby("signup_date").size().rename("new_signups").reset_index().rename(
        columns={"signup_date": "Date"}
    )
    out = pd.DataFrame({"Date": all_dates}).merge(daily, on="Date", how="left")
    out["new_signups"] = out["new_signups"].fillna(0)
    out["signups_rmean28"] = out["new_signups"].rolling(28, min_periods=1).mean()
    return out


# ---------------------------------------------------------------------------
# Master build
# ---------------------------------------------------------------------------


def build_exog_v4(all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds the full v4 daily exogenous feature frame (no lag/rolling yet).

    Columns in `LEAKY_LEVEL_COLS_V4` are level-leaky and must be handled
    by the model runner with DoY-mean imputation in both train and horizon.
    """
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(DATA / "order_items.csv")
    products = pd.read_csv(DATA / "products.csv")
    payments = pd.read_csv(DATA / "payments.csv")
    returns = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
    reviews = pd.read_csv(DATA / "reviews.csv", parse_dates=["review_date"])
    shipments = pd.read_csv(DATA / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
    web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
    promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
    geography = pd.read_csv(DATA / "geography.csv")

    base = pd.DataFrame({"Date": all_dates})

    orders_agg = _agg_orders_v4(orders)
    items_agg = _agg_items_by_mix(orders, items, products)
    pay_agg = _agg_payments_v4(payments, orders)
    ret_order_agg = _agg_returns_on_order(returns, orders)
    rev_order_agg = _agg_reviews_on_order(reviews, orders)
    ship_agg = _agg_shipments_v4(shipments)
    web_agg = _agg_web_v4(web)
    promo_agg = _agg_promotions_v4(promotions, all_dates)
    cust_new_agg = _agg_customers_new(customers, all_dates)
    inv_agg = _agg_inventory_v4(inventory, all_dates)
    geo_agg = _agg_geography(orders, geography)
    cust_life_agg = _agg_customer_lifecycle(orders, customers, all_dates)

    for part in (
        orders_agg, items_agg, pay_agg, ret_order_agg, rev_order_agg,
        ship_agg, web_agg, cust_new_agg, geo_agg, cust_life_agg,
    ):
        base = base.merge(part, on="Date", how="left")
    base = base.merge(promo_agg, on="Date", how="left")
    base = base.merge(inv_agg, on="Date", how="left")

    # Derived ratios (module B)
    oc = base["orders_count"].replace(0, np.nan)
    base["basket_size"] = base["items_total_qty_v4"] / oc
    base["avg_order_value"] = base["items_gross_value"] / oc
    gv_nz = base["items_gross_value"].replace(0, np.nan)
    base["discount_rate"] = base["items_discount_total"] / gv_nz
    base["cogs_to_gross_ratio"] = base["items_cogs_total_value"] / gv_nz

    # Clean up
    for c in base.columns:
        if c == "Date":
            continue
        if base[c].dtype == bool:
            base[c] = base[c].astype(int)
    return base


if __name__ == "__main__":
    dates = pd.date_range("2012-07-04", "2024-07-01", freq="D")
    print(f"Building v4 exogenous frame for {len(dates)} days ...")
    df = build_exog_v4(dates)
    print(f"Frame shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print()
    # Quick NaN audit on training period
    train = df[df["Date"].between("2014-01-01", "2022-12-31")]
    nan_rates = train.drop(columns=["Date"]).isna().mean().sort_values(ascending=False).head(15)
    print("Top-15 NaN rates on 2014-2022:")
    print(nan_rates)
