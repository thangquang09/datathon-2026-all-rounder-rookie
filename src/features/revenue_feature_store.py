"""Time-aware daily feature store for Revenue and COGS forecasting.

The competition target is daily `Revenue` for 2023-01-01 to 2024-07-01,
and the submission file also contains `COGS`. This module therefore
builds one row per `Date` and avoids same-day future transaction signals.

Leakage policy:
- Target-derived lag/rolling features are computed from values strictly
  before the forecast date. During inference they can be updated
  recursively with prior predictions.
- Transactional, web, inventory, promotion, review, return, shipment,
  payment, product, and customer signals are never used at same-day level.
  They are converted to historical lags, two-year lags, or shifted rolling
  averages. This is important because the test period has no ground-truth
  orders/web/inventory/returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.calendar_vn import add_vn_calendar


VALID_ORDER_STATUS = ("paid", "shipped", "delivered", "returned")
TARGET_COLUMNS = ("Revenue", "COGS")


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Configuration for the daily forecasting feature store.

    Attributes:
        data_dir: Directory containing competition CSV files.
        forecast_start: First date to forecast.
        forecast_end: Last date to forecast.
        use_lag365_exogenous: Keep one-year exogenous lag features. These
            are useful for 2023 but become missing for much of 2024 unless
            the underlying exogenous series is also forecast. Keeping them
            is safe because missing values are imputed by models; use
            lag730 and seasonal aggregates as more robust future signals.
    """

    data_dir: Path | str = Path("data")
    forecast_start: str = "2023-01-01"
    forecast_end: str = "2024-07-01"
    use_lag365_exogenous: bool = True
    use_short_exogenous_lags: bool = False
    include_external_vn_calendar: bool = False
    include_known_promo_calendar: bool = False


def _read_csv(data_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, low_memory=False, **kwargs)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
        .strip("_")
        for c in out.columns
    ]
    return out


def _dense_daily(all_dates: pd.DatetimeIndex, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Date": all_dates})
    if df.empty:
        return out
    tmp = df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    return out.merge(tmp, on="Date", how="left")


def _share_table(
    df: pd.DataFrame,
    date_col: str,
    category_col: str,
    value_col: str | None,
    prefix: str,
    allowed_values: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date"])
    cols = [date_col, category_col] + ([value_col] if value_col is not None else [])
    tmp = df[cols].dropna(subset=[date_col]).copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    if allowed_values is not None:
        tmp = tmp[tmp[category_col].isin(list(allowed_values))]
    if value_col is None:
        pivot = tmp.groupby([date_col, category_col]).size().unstack(fill_value=0)
    else:
        pivot = tmp.pivot_table(index=date_col, columns=category_col, values=value_col, aggfunc="sum", fill_value=0)
    totals = pivot.sum(axis=1).replace(0, np.nan)
    share = pivot.div(totals, axis=0).fillna(0)
    share.columns = [f"{prefix}_{str(c).lower().replace('-', '_')}_share" for c in share.columns]
    return share.reset_index().rename(columns={date_col: "Date"})


def _entropy_table(df: pd.DataFrame, date_col: str, category_col: str, feature_name: str) -> pd.DataFrame:
    if df.empty or category_col not in df.columns:
        return pd.DataFrame(columns=["Date", feature_name])
    tmp = df[[date_col, category_col]].dropna(subset=[date_col, category_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["Date", feature_name])
    pivot = tmp.groupby([date_col, category_col]).size().unstack(fill_value=0)
    share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    entropy = -(share * np.log(share.replace(0, np.nan))).sum(axis=1).fillna(0)
    return entropy.reset_index(name=feature_name).rename(columns={date_col: "Date"})


def _add_lagged_exogenous_features(
    daily: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    prefix: str,
    include_lag365: bool = True,
    include_short_lags: bool = False,
) -> pd.DataFrame:
    """Convert same-day exogenous daily metrics to future-safe features.

    For a date t, `*_lag730` uses information from two years earlier,
    available throughout the 2023-2024 test horizon. `*_roll365_lag365`
    means the average of historical values ending one year before t, so it
    does not require actual 2023-2024 exogenous observations.
    """

    dense = _dense_daily(all_dates, daily)
    metric_cols = [c for c in dense.columns if c != "Date"]
    features: dict[str, pd.Series | pd.DatetimeIndex] = {"Date": all_dates}
    for col in metric_cols:
        s = dense[col]
        if include_short_lags:
            for lag in (7, 14, 28):
                features[f"{prefix}_{col}_lag{lag}"] = s.shift(lag)
            features[f"{prefix}_{col}_roll7_lag1"] = s.shift(1).rolling(7, min_periods=3).mean()
            features[f"{prefix}_{col}_roll28_lag1"] = s.shift(1).rolling(28, min_periods=7).mean()
        if include_lag365:
            features[f"{prefix}_{col}_lag365"] = s.shift(365)
        features[f"{prefix}_{col}_lag730"] = s.shift(730)
        features[f"{prefix}_{col}_roll365_lag365"] = s.shift(365).rolling(365, min_periods=30).mean()
    return pd.DataFrame(features)


def _add_calendar_features(frame: pd.DataFrame, include_external_vn_calendar: bool = False) -> pd.DataFrame:
    out = frame.copy()
    d = pd.to_datetime(out["Date"])
    out["day_of_week"] = d.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_month"] = d.dt.day
    out["day_of_year"] = d.dt.dayofyear
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_quarter_start"] = d.dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    out["is_year_start"] = d.dt.is_year_start.astype(int)
    out["is_year_end"] = d.dt.is_year_end.astype(int)
    out["days_to_month_end"] = d.dt.days_in_month - d.dt.day
    out["days_from_month_start"] = d.dt.day - 1
    out["days_from_year_start"] = d.dt.dayofyear - 1
    out["days_until_year_end"] = pd.to_datetime(d.dt.year.astype(str) + "-12-31").sub(d).dt.days
    out["is_first_week_of_month"] = (d.dt.day <= 7).astype(int)
    out["is_last_week_of_month"] = (out["days_to_month_end"] <= 6).astype(int)
    out["is_mid_month"] = d.dt.day.between(12, 18).astype(int)
    out["time_index"] = (d - d.min()).dt.days
    out["time_index_squared"] = out["time_index"] ** 2
    out["sin_day_of_week"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["cos_day_of_week"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)
    out["sin_month_1"] = out["sin_month"]
    out["cos_month_1"] = out["cos_month"]
    out["sin_day_of_year"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_day_of_year"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["is_near_new_year"] = ((d.dt.month == 1) & (d.dt.day <= 7)).astype(int)
    out["is_mid_year_period"] = d.dt.month.isin([6, 7]).astype(int)
    out["is_fall_period"] = d.dt.month.isin([8, 9, 10]).astype(int)
    out["is_year_end_period"] = d.dt.month.isin([11, 12]).astype(int)
    out["is_payday_proxy_25_31"] = (d.dt.day >= 25).astype(int)
    out["is_payday_proxy_1_5"] = (d.dt.day <= 5).astype(int)
    if include_external_vn_calendar:
        out = add_vn_calendar(out, date_col="Date")
    return _sanitize_columns(out)


def _build_orders_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    geography = data["geography"].copy()
    customers = data["customers"].copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    orders = orders.merge(geography[["zip", "region", "city", "district"]], on="zip", how="left")
    orders = orders.merge(
        customers[["customer_id", "signup_date", "gender", "age_group", "acquisition_channel"]],
        on="customer_id",
        how="left",
    )
    orders["is_valid_order"] = orders["order_status"].isin(VALID_ORDER_STATUS).astype(int)
    orders["is_delivered"] = (orders["order_status"] == "delivered").astype(int)
    orders["is_cancelled"] = (orders["order_status"] == "cancelled").astype(int)
    orders["is_returned_status"] = (orders["order_status"] == "returned").astype(int)
    orders["is_mobile"] = (orders["device_type"] == "mobile").astype(int)
    orders["is_desktop"] = (orders["device_type"] == "desktop").astype(int)
    orders["is_tablet"] = (orders["device_type"] == "tablet").astype(int)
    orders["customer_tenure_days"] = (orders["order_date"] - orders["signup_date"]).dt.days.clip(lower=0)
    daily = (
        orders.groupby("order_date")
        .agg(
            n_orders=("order_id", "nunique"),
            n_valid_orders=("is_valid_order", "sum"),
            n_unique_customers=("customer_id", "nunique"),
            delivered_rate=("is_delivered", "mean"),
            cancelled_rate=("is_cancelled", "mean"),
            returned_status_rate=("is_returned_status", "mean"),
            mobile_share=("is_mobile", "mean"),
            desktop_share=("is_desktop", "mean"),
            tablet_share=("is_tablet", "mean"),
            unique_cities=("city", "nunique"),
            unique_districts=("district", "nunique"),
            avg_customer_tenure_days=("customer_tenure_days", "mean"),
            median_customer_tenure_days=("customer_tenure_days", "median"),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    city_counts = orders.groupby(["order_date", "city"]).size()
    daily_city = (
        city_counts.groupby(level=0).max()
        .div(city_counts.groupby(level=0).sum().replace(0, np.nan))
        .reset_index(name="top_city_share")
        .rename(columns={"order_date": "Date"})
    )
    district_counts = orders.groupby(["order_date", "district"]).size()
    daily_district = (
        district_counts.groupby(level=0).max()
        .div(district_counts.groupby(level=0).sum().replace(0, np.nan))
        .reset_index(name="top_district_share")
        .rename(columns={"order_date": "Date"})
    )
    source_share = _share_table(orders, "order_date", "order_source", None, "source")
    payment_share = _share_table(orders, "order_date", "payment_method", None, "payment_method")
    region_share = _share_table(orders, "order_date", "region", None, "region")
    gender_share = _share_table(orders, "order_date", "gender", None, "gender")
    age_share = _share_table(orders, "order_date", "age_group", None, "age_group")
    acq_share = _share_table(orders, "order_date", "acquisition_channel", None, "acq_channel")
    source_entropy = _entropy_table(orders, "order_date", "order_source", "source_entropy")
    device_entropy = _entropy_table(orders, "order_date", "device_type", "device_entropy")
    payment_entropy = _entropy_table(orders, "order_date", "payment_method", "payment_entropy")
    for part in (
        daily_city,
        daily_district,
        source_share,
        payment_share,
        region_share,
        gender_share,
        age_share,
        acq_share,
        source_entropy,
        device_entropy,
        payment_entropy,
    ):
        daily = daily.merge(part, on="Date", how="left")
    return _sanitize_columns(daily)


def _build_items_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    items = data["order_items"].copy()
    products = data["products"].copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders = orders[orders["order_status"].isin(VALID_ORDER_STATUS)]
    products["product_margin_rate"] = _safe_div(products["price"] - products["cogs"], products["price"])
    products["price_bucket"] = pd.qcut(
        products["price"].rank(method="first"),
        q=3,
        labels=["low", "mid", "high"],
    ).astype(str)
    products["margin_bucket"] = pd.qcut(
        products["product_margin_rate"].rank(method="first"),
        q=3,
        labels=["low", "mid", "high"],
    ).astype(str)
    df = (
        items.merge(orders[["order_id", "order_date"]], on="order_id", how="inner")
        .merge(
            products[
                [
                    "product_id",
                    "category",
                    "segment",
                    "size",
                    "color",
                    "price",
                    "cogs",
                    "product_margin_rate",
                    "price_bucket",
                    "margin_bucket",
                ]
            ],
            on="product_id",
            how="left",
        )
    )
    df["line_revenue"] = df["quantity"] * df["unit_price"]
    df["line_cogs"] = df["quantity"] * df["cogs"]
    df["line_profit"] = df["line_revenue"] - df["line_cogs"]
    df["promo_line"] = df[["promo_id", "promo_id_2"]].notna().any(axis=1).astype(int)
    df["double_promo_line"] = df[["promo_id", "promo_id_2"]].notna().all(axis=1).astype(int)
    daily = (
        df.groupby("order_date")
        .agg(
            total_units=("quantity", "sum"),
            n_item_orders=("order_id", "nunique"),
            item_lines=("product_id", "size"),
            mean_qty_per_line=("quantity", "mean"),
            mean_unit_price=("unit_price", "mean"),
            revenue_from_items=("line_revenue", "sum"),
            cogs_from_items=("line_cogs", "sum"),
            gross_profit_from_items=("line_profit", "sum"),
            total_discount=("discount_amount", "sum"),
            promo_usage_rate=("promo_line", "mean"),
            double_promo_rate=("double_promo_line", "mean"),
            unique_products=("product_id", "nunique"),
            avg_product_margin_rate=("product_margin_rate", "mean"),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    daily["items_per_order"] = _safe_div(daily["total_units"], daily["n_item_orders"])
    daily["weighted_avg_unit_price"] = _safe_div(daily["revenue_from_items"], daily["total_units"])
    daily["item_margin_rate"] = _safe_div(daily["gross_profit_from_items"], daily["revenue_from_items"])
    daily["revenue_weighted_margin_rate"] = daily["item_margin_rate"]
    daily["discount_rate"] = _safe_div(daily["total_discount"], daily["revenue_from_items"] + daily["total_discount"])
    product_revenue = df.groupby(["order_date", "product_id"])["line_revenue"].sum()
    product_share = product_revenue.div(product_revenue.groupby(level=0).sum().replace(0, np.nan))
    concentration = (
        pd.DataFrame(
            {
                "product_hhi": product_share.pow(2).groupby(level=0).sum(),
                "top_product_share": product_share.groupby(level=0).max(),
            }
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    daily = daily.merge(concentration, on="Date", how="left")
    for part in (
        _share_table(df, "order_date", "category", "line_revenue", "category"),
        _share_table(df, "order_date", "segment", "line_revenue", "segment"),
        _share_table(df, "order_date", "size", "line_revenue", "size"),
        _share_table(df, "order_date", "color", "line_revenue", "color"),
        _share_table(df, "order_date", "price_bucket", "line_revenue", "price_bucket"),
        _share_table(df, "order_date", "margin_bucket", "line_revenue", "margin_bucket"),
    ):
        daily = daily.merge(part, on="Date", how="left")
    return _sanitize_columns(daily)


def _build_payments_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    payments = data["payments"].copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders = orders[orders["order_status"].isin(VALID_ORDER_STATUS)]
    df = payments.merge(orders[["order_id", "order_date", "payment_method"]], on="order_id", how="inner")
    df["is_installment"] = (df["installments"] > 1).astype(int)
    daily = (
        df.groupby("order_date")
        .agg(
            total_payment_value=("payment_value", "sum"),
            mean_payment_value=("payment_value", "mean"),
            median_payment_value=("payment_value", "median"),
            std_payment_value=("payment_value", "std"),
            mean_installments=("installments", "mean"),
            installment_ratio=("is_installment", "mean"),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    return _sanitize_columns(daily)


def _build_shipments_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    shipments = data["shipments"].copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    shipments["ship_date"] = pd.to_datetime(shipments["ship_date"])
    shipments["delivery_date"] = pd.to_datetime(shipments["delivery_date"])
    df = shipments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    df["transit_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
    df["fulfill_days"] = (df["delivery_date"] - df["order_date"]).dt.days
    df["free_shipping"] = (df["shipping_fee"] == 0).astype(int)
    daily = (
        df.groupby("order_date")
        .agg(
            mean_shipping_fee=("shipping_fee", "mean"),
            total_shipping_fee=("shipping_fee", "sum"),
            free_shipping_share=("free_shipping", "mean"),
            mean_transit_days=("transit_days", "mean"),
            mean_fulfill_days=("fulfill_days", "mean"),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    return _sanitize_columns(daily)


def _build_returns_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = data["returns"].copy()
    returns["return_date"] = pd.to_datetime(returns["return_date"])
    daily = (
        returns.groupby("return_date")
        .agg(
            n_returns=("return_id", "nunique"),
            total_return_qty=("return_quantity", "sum"),
            total_refund=("refund_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"return_date": "Date"})
    )
    reason_share = _share_table(returns, "return_date", "return_reason", None, "return_reason")
    daily = daily.merge(reason_share, on="Date", how="left")
    return _sanitize_columns(daily)


def _build_reviews_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reviews = data["reviews"].copy()
    reviews["review_date"] = pd.to_datetime(reviews["review_date"])
    reviews["is_negative"] = (reviews["rating"] <= 2).astype(int)
    reviews["is_5star"] = (reviews["rating"] == 5).astype(int)
    daily = (
        reviews.groupby("review_date")
        .agg(
            n_reviews=("review_id", "nunique"),
            mean_rating=("rating", "mean"),
            pct_negative=("is_negative", "mean"),
            pct_5star=("is_5star", "mean"),
        )
        .reset_index()
        .rename(columns={"review_date": "Date"})
    )
    return _sanitize_columns(daily)


def _build_web_daily(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    web = data["web_traffic"].copy()
    web["date"] = pd.to_datetime(web["date"])
    web["pages_per_session"] = _safe_div(web["page_views"], web["sessions"])
    web["sessions_per_visitor"] = _safe_div(web["sessions"], web["unique_visitors"])
    web["engaged_sessions"] = web["sessions"] * (1 - web["bounce_rate"])
    daily = (
        web.groupby("date")
        .agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration_sec=("avg_session_duration_sec", "mean"),
            pages_per_session=("pages_per_session", "mean"),
            sessions_per_visitor=("sessions_per_visitor", "mean"),
            engaged_sessions=("engaged_sessions", "sum"),
        )
        .reset_index()
        .rename(columns={"date": "Date"})
    )
    source_sessions = _share_table(web, "date", "traffic_source", "sessions", "traffic_source")
    daily = daily.merge(source_sessions, on="Date", how="left")
    return _sanitize_columns(daily)


def _build_inventory_asof(data: dict[str, pd.DataFrame], all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    inventory = data["inventory"].copy()
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"])
    daily = (
        inventory.groupby("snapshot_date")
        .agg(
            stock_on_hand=("stock_on_hand", "sum"),
            units_received=("units_received", "sum"),
            units_sold=("units_sold", "sum"),
            stockout_rate=("stockout_flag", "mean"),
            overstock_rate=("overstock_flag", "mean"),
            reorder_rate=("reorder_flag", "mean"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_days_of_supply=("days_of_supply", "mean"),
            median_days_of_supply=("days_of_supply", "median"),
            mean_sell_through=("sell_through_rate", "mean"),
            n_stockout_products=("stockout_flag", "sum"),
            low_fill_rate_share=("fill_rate", lambda s: (s < 0.9).mean()),
            high_sell_through_share=("sell_through_rate", lambda s: (s > 0.7).mean()),
        )
        .reset_index()
    )
    daily["overstock_pressure_index"] = daily["overstock_rate"] * daily["mean_days_of_supply"]
    daily["stock_to_sales_ratio"] = _safe_div(daily["stock_on_hand"], daily["units_sold"])
    # Month-end snapshot becomes known from the next day onward.
    daily["effective_date"] = daily["snapshot_date"] + pd.Timedelta(days=1)
    daily = daily.sort_values("effective_date")
    base = pd.DataFrame({"Date": all_dates})
    merged = pd.merge_asof(base.sort_values("Date"), daily.drop(columns=["snapshot_date"]), left_on="Date", right_on="effective_date", direction="backward")
    merged = merged.drop(columns=["effective_date"])
    metric_cols = [c for c in merged.columns if c != "Date"]
    for col in metric_cols:
        merged[f"{col}_mom"] = merged[col] - merged[col].shift(31)
        merged[f"{col}_yoy"] = merged[col] - merged[col].shift(365)
    return _sanitize_columns(merged)


def _build_promo_calendar(data: dict[str, pd.DataFrame], all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    promotions = data["promotions"].copy()
    promotions["start_date"] = pd.to_datetime(promotions["start_date"])
    promotions["end_date"] = pd.to_datetime(promotions["end_date"])
    rows = []
    for date in all_dates:
        active = promotions[(promotions["start_date"] <= date) & (promotions["end_date"] >= date)]
        rows.append(
            {
                "Date": date,
                "promo_n_active_known": len(active),
                "promo_has_active_known": int(len(active) > 0),
                "promo_mean_discount_value_known": active["discount_value"].mean() if len(active) else 0,
                "promo_max_discount_value_known": active["discount_value"].max() if len(active) else 0,
                "promo_is_percentage_known": int((active["promo_type"] == "percentage").any()) if len(active) else 0,
                "promo_has_stackable_known": int((active["stackable_flag"] == 1).any()) if len(active) else 0,
            }
        )
    out = pd.DataFrame(rows)
    d = pd.to_datetime(out["Date"])
    # Calendar proxies are known for test and mirror historical sale windows
    # described by the provided promotion names, not external data.
    out["is_spring_sale_proxy"] = ((d.dt.month == 3) | ((d.dt.month == 4) & (d.dt.day <= 17))).astype(int)
    out["is_midyear_sale_proxy"] = d.dt.month.isin([6, 7]).astype(int)
    out["is_fall_launch_proxy"] = d.dt.month.isin([8, 9, 10]).astype(int)
    out["is_yearend_sale_proxy"] = d.dt.month.isin([11, 12]).astype(int)
    return _sanitize_columns(out)


def _load_competition_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "sales": _read_csv(data_dir, "sales.csv", parse_dates=["Date"]),
        "sample_submission": _read_csv(data_dir, "sample_submission.csv", parse_dates=["Date"]),
        "orders": _read_csv(data_dir, "orders.csv"),
        "order_items": _read_csv(data_dir, "order_items.csv"),
        "products": _read_csv(data_dir, "products.csv"),
        "customers": _read_csv(data_dir, "customers.csv", parse_dates=["signup_date"]),
        "geography": _read_csv(data_dir, "geography.csv"),
        "payments": _read_csv(data_dir, "payments.csv"),
        "shipments": _read_csv(data_dir, "shipments.csv"),
        "returns": _read_csv(data_dir, "returns.csv"),
        "reviews": _read_csv(data_dir, "reviews.csv"),
        "inventory": _read_csv(data_dir, "inventory.csv"),
        "web_traffic": _read_csv(data_dir, "web_traffic.csv"),
        "promotions": _read_csv(data_dir, "promotions.csv"),
    }


def _days_since_shifted_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    shifted = series.shift(1)
    values = shifted.to_numpy()
    out = np.full(len(series), np.nan)
    reducer = np.nanargmax if mode == "max" else np.nanargmin
    for i in range(len(series)):
        start = max(0, i - window)
        arr = values[start:i]
        valid = arr[~np.isnan(arr)]
        if len(valid) < max(7, window // 4):
            continue
        segment = arr.copy()
        if np.isnan(segment).all():
            continue
        local_idx = reducer(segment)
        out[i] = i - (start + local_idx)
    return pd.Series(out, index=series.index)


def add_dynamic_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add target lag/rolling features using values before each date.

    This function can be called repeatedly during recursive forecasting
    after filling previous test-date predictions.
    """

    out = frame.sort_values("Date").copy()
    features: dict[str, pd.Series] = {}
    for target in TARGET_COLUMNS:
        name = target.lower()
        s = out[target]
        for lag in (1, 2, 3, 7, 14, 21, 28, 30, 56, 60, 90, 91, 182, 365, 366, 728, 730):
            features[f"{name}_lag_{lag}"] = s.shift(lag)
        shifted = s.shift(1)
        for window in (3, 7, 14, 28, 91, 182, 365):
            features[f"{name}_roll_mean_{window}"] = shifted.rolling(window, min_periods=max(2, min(window, 14))).mean()
            features[f"{name}_roll_median_{window}"] = shifted.rolling(window, min_periods=max(2, min(window, 14))).median()
            features[f"{name}_roll_std_{window}"] = shifted.rolling(window, min_periods=max(3, min(window, 14))).std()
        for window in (7, 28, 91):
            features[f"{name}_roll_min_{window}"] = shifted.rolling(window, min_periods=max(3, min(window, 14))).min()
            features[f"{name}_roll_max_{window}"] = shifted.rolling(window, min_periods=max(3, min(window, 14))).max()
        features[f"{name}_roll_q25_28"] = shifted.rolling(28, min_periods=7).quantile(0.25)
        features[f"{name}_roll_q75_28"] = shifted.rolling(28, min_periods=7).quantile(0.75)
        features[f"{name}_ewm_7"] = shifted.ewm(span=7, adjust=False, min_periods=3).mean()
        features[f"{name}_ewm_28"] = shifted.ewm(span=28, adjust=False, min_periods=7).mean()
        features[f"{name}_ewm_90"] = shifted.ewm(span=90, adjust=False, min_periods=14).mean()
        features[f"{name}_same_weekday_lastyear"] = s.shift(364)
        features[f"{name}_same_weekday_2yr"] = s.shift(728)
        features[f"{name}_same_weekday_4w_mean"] = pd.concat(
            [s.shift(7 * k) for k in range(1, 5)],
            axis=1,
        ).mean(axis=1)
        features[f"{name}_same_weekday_8w_mean"] = pd.concat(
            [s.shift(7 * k) for k in range(1, 9)],
            axis=1,
        ).mean(axis=1)
        features[f"{name}_same_dayofyear_3y_mean"] = pd.concat(
            [s.shift(365), s.shift(730), s.shift(1095)],
            axis=1,
        ).mean(axis=1)
        features[f"{name}_diff_1"] = s.shift(1) - s.shift(2)
        features[f"{name}_diff_7"] = s.shift(1) - s.shift(8)
        features[f"{name}_pct_change_1"] = _safe_div(s.shift(1), s.shift(2)) - 1
        features[f"{name}_pct_change_7"] = _safe_div(s.shift(1), s.shift(8)) - 1
        features[f"{name}_yoy_change"] = _safe_div(s.shift(365), s.shift(730)) - 1
        features[f"{name}_mom_proxy"] = _safe_div(s.shift(30), s.shift(60)) - 1
        features[f"{name}_zscore_28"] = _safe_div(
            s.shift(1) - features[f"{name}_roll_mean_28"],
            features[f"{name}_roll_std_28"],
        )
        features[f"{name}_days_since_rolling_peak_91"] = _days_since_shifted_extreme(s, 91, "max")
        features[f"{name}_days_since_rolling_trough_91"] = _days_since_shifted_extreme(s, 91, "min")
    target_features = pd.DataFrame(features, index=out.index)
    target_features["gross_profit_lag_1"] = target_features["revenue_lag_1"] - target_features["cogs_lag_1"]
    target_features["gross_profit_lag_365"] = target_features["revenue_lag_365"] - target_features["cogs_lag_365"]
    target_features["gross_margin_lag_1"] = _safe_div(target_features["gross_profit_lag_1"], target_features["revenue_lag_1"])
    target_features["gross_margin_lag_365"] = _safe_div(target_features["gross_profit_lag_365"], target_features["revenue_lag_365"])
    target_features["cogs_ratio_lag_365"] = _safe_div(target_features["cogs_lag_365"], target_features["revenue_lag_365"])
    target_features["revenue_to_cogs_ratio_lag_365"] = _safe_div(target_features["revenue_lag_365"], target_features["cogs_lag_365"])
    existing_dynamic = [c for c in out.columns if c in target_features.columns]
    out = out.drop(columns=existing_dynamic, errors="ignore")
    return pd.concat([out, target_features], axis=1)


def _add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "revenue_lag_365" in out.columns:
        out["weekend_x_revenue_lag365"] = out["is_weekend"] * out["revenue_lag_365"]
        out["month_x_revenue_lag365"] = out["month"] * out["revenue_lag_365"]
    if {"wt_sessions_lag365", "wt_pages_per_session_lag365"}.issubset(out.columns):
        out["sessions_x_pages_per_session_lag365"] = out["wt_sessions_lag365"] * out["wt_pages_per_session_lag365"]
    if {"wt_sessions_lag365", "cvr_conversion_rate_lag365"}.issubset(out.columns):
        out["sessions_x_cvr_lag365"] = out["wt_sessions_lag365"] * out["cvr_conversion_rate_lag365"]
    if {"stockout_rate", "revenue_lag_365"}.issubset(out.columns):
        out["stockout_rate_x_revenue_lag365"] = out["stockout_rate"] * out["revenue_lag_365"]
    if {"mean_fill_rate", "revenue_lag_365"}.issubset(out.columns):
        out["fill_rate_x_revenue_lag365"] = out["mean_fill_rate"] * out["revenue_lag_365"]
    if {"oi_discount_rate_lag365", "oi_promo_usage_rate_lag365"}.issubset(out.columns):
        out["discount_rate_x_promo_usage_lag365"] = out["oi_discount_rate_lag365"] * out["oi_promo_usage_rate_lag365"]
    if {"rev_mean_rating_lag365", "rev_n_reviews_lag365"}.issubset(out.columns):
        out["rating_x_review_count_lag365"] = out["rev_mean_rating_lag365"] * out["rev_n_reviews_lag365"]
    if {"shp_mean_fulfill_days_lag365", "rev_pct_negative_lag365"}.issubset(out.columns):
        out["delivery_delay_x_negative_review_lag365"] = out["shp_mean_fulfill_days_lag365"] * out["rev_pct_negative_lag365"]
    if {"ret_n_returns_lag365", "oi_category_streetwear_share_lag365"}.issubset(out.columns):
        out["return_count_x_streetwear_share_lag365"] = out["ret_n_returns_lag365"] * out["oi_category_streetwear_share_lag365"]
    return out


def build_revenue_feature_store(config: FeatureStoreConfig | None = None) -> pd.DataFrame:
    """Build a time-aware daily feature store for revenue and COGS models."""

    cfg = config or FeatureStoreConfig()
    data_dir = Path(cfg.data_dir)
    data = _load_competition_data(data_dir)
    sales = data["sales"].sort_values("Date").copy()
    sample = data["sample_submission"].sort_values("Date").copy()
    all_dates = pd.date_range(sales["Date"].min(), sample["Date"].max(), freq="D")

    frame = pd.DataFrame({"Date": all_dates})
    frame = frame.merge(sales[["Date", "Revenue", "COGS"]], on="Date", how="left")
    frame = _add_calendar_features(frame, include_external_vn_calendar=cfg.include_external_vn_calendar)

    exog_builders = {
        "ord": _build_orders_daily,
        "oi": _build_items_daily,
        "pay": _build_payments_daily,
        "shp": _build_shipments_daily,
        "ret": _build_returns_daily,
        "rev": _build_reviews_daily,
        "wt": _build_web_daily,
    }
    for prefix, builder in exog_builders.items():
        daily = builder(data)
        hist = _add_lagged_exogenous_features(
            daily=daily,
            all_dates=all_dates,
            prefix=prefix,
            include_lag365=cfg.use_lag365_exogenous,
            include_short_lags=cfg.use_short_exogenous_lags,
        )
        frame = frame.merge(hist, on="Date", how="left")

    inventory = _build_inventory_asof(data, all_dates)
    frame = frame.merge(inventory, on="Date", how="left")
    if cfg.include_known_promo_calendar:
        promotions = _build_promo_calendar(data, all_dates)
        frame = frame.merge(promotions, on="Date", how="left")

    frame = add_dynamic_target_features(frame)
    frame = _add_interaction_features(frame)
    frame = _sanitize_columns(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame
