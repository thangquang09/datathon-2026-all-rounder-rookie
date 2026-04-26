"""Segment behavior profiling and visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .rfm_objective_segmentation import FINAL_SEGMENT_ORDER


DEFAULT_RADAR_METRICS: Mapping[str, Tuple[str, bool]] = {
    "Profit": ("avg_profit", True),
    "Frequency": ("avg_frequency", True),
    "Recency health": ("avg_recency", False),
    "Margin": ("avg_margin", True),
    "Low promo dependency": ("promo_usage", False),
    "Rating": ("avg_rating", True),
    "Low refund rate": ("refund_rate", False),
}


@dataclass(frozen=True)
class SegmentBehaviorProfiles:
    """Segment-level behavior matrices and long-form top tables."""

    share_matrices: Dict[str, pd.DataFrame]
    top_tables: Dict[str, pd.DataFrame]


def _normalize_metric(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median() if numeric.notna().any() else 0)
    low = numeric.min()
    high = numeric.max()
    if np.isclose(high, low):
        scaled = pd.Series(0.5, index=values.index)
    else:
        scaled = (numeric - low) / (high - low)
    return scaled if higher_is_better else 1 - scaled


def plot_segment_radar(
    segment_summary: pd.DataFrame,
    segment_order: Sequence[str] = FINAL_SEGMENT_ORDER,
    metric_specs: Mapping[str, Tuple[str, bool]] = DEFAULT_RADAR_METRICS,
    figsize: Tuple[int, int] = (10, 8),
):
    """Plot a radar chart describing the six final segment labels.

    Metrics are normalized across the segment summary itself, so the chart is a
    relative behavior profile rather than an absolute score. Metrics where lower
    is better, such as recency, promo usage, and refund rate, are inverted so a
    larger radar radius consistently means a more desirable behavior signal.
    """
    import matplotlib.pyplot as plt

    summary = segment_summary.copy()
    if summary.index.name is None and "final_segment" in summary.columns:
        summary = summary.set_index("final_segment")

    ordered_segments = [segment for segment in segment_order if segment in summary.index]
    metrics = {
        label: (column, higher_is_better)
        for label, (column, higher_is_better) in metric_specs.items()
        if column in summary.columns
    }
    if not ordered_segments or not metrics:
        raise ValueError("segment_summary must contain segment rows and at least one radar metric.")

    normalized = pd.DataFrame(index=ordered_segments)
    for label, (column, higher_is_better) in metrics.items():
        normalized[label] = _normalize_metric(summary.loc[ordered_segments, column], higher_is_better)

    labels = list(normalized.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    colors = ["#0b6e4f", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8d99ae"]

    for idx, segment in enumerate(ordered_segments):
        values = normalized.loc[segment].tolist()
        values += values[:1]
        ax.plot(angles, values, label=segment, color=colors[idx % len(colors)], linewidth=2)
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Final Segment Behavior Radar", pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), title="Final segment")
    plt.tight_layout()
    return fig


def _share_matrix(
    df: pd.DataFrame,
    segment_col: str,
    category_col: str,
    segment_order: Sequence[str],
    top_n: int = 6,
    weight_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = df[[segment_col, category_col] + ([weight_col] if weight_col else [])].copy()
    data = data.dropna(subset=[segment_col, category_col])
    data[category_col] = data[category_col].astype(str)

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if weight_col:
        grouped = data.groupby([segment_col, category_col], as_index=False)[weight_col].sum()
        grouped = grouped.rename(columns={weight_col: "value"})
    else:
        grouped = data.groupby([segment_col, category_col], as_index=False).size()
        grouped = grouped.rename(columns={"size": "value"})

    totals = grouped.groupby(segment_col)["value"].transform("sum")
    grouped["share"] = grouped["value"] / totals

    ordered_segments = [segment for segment in segment_order if segment in grouped[segment_col].unique()]
    top_categories = []
    for segment in ordered_segments:
        top_categories.extend(
            grouped[grouped[segment_col] == segment]
            .nlargest(top_n, "value")[category_col]
            .tolist()
        )
    top_categories = list(dict.fromkeys(top_categories))

    matrix = (
        grouped[grouped[category_col].isin(top_categories)]
        .pivot(index=segment_col, columns=category_col, values="share")
        .reindex(index=ordered_segments, columns=top_categories)
        .fillna(0)
    )
    top_table = grouped.sort_values([segment_col, "value"], ascending=[True, False])
    return matrix, top_table


def build_segment_behavior_profiles(
    customer_table: pd.DataFrame,
    data_dir: Path,
    segment_col: str = "final_segment",
    segment_order: Sequence[str] = FINAL_SEGMENT_ORDER,
    top_n: int = 6,
) -> SegmentBehaviorProfiles:
    """Build behavior profiles for product, place, promo, order, and review signals.

    The function uses `customer_table` for final segment labels, then joins the
    original transaction tables in `data_dir` to recover details that are not in
    the golden table: payment method, device type, product segment/name,
    promotion campaign, promotion channel, and reviewed product category.
    Shares are row-normalized within each segment so the heatmaps answer "what
    does this segment over-index on?" instead of merely showing which segment is
    largest.
    """
    data_path = Path(data_dir)
    customer_segments = customer_table[["customer_id", segment_col]].drop_duplicates()
    customer_segments = customer_segments.dropna(subset=[segment_col])

    orders = pd.read_csv(data_path / "orders.csv")
    order_items = pd.read_csv(data_path / "order_items.csv", low_memory=False)
    products = pd.read_csv(data_path / "products.csv")
    product_lookup = products[["product_id", "product_name", "category", "segment"]].rename(
        columns={"segment": "product_segment"}
    )
    review_product_lookup = products[["product_id", "product_name", "category", "segment"]].rename(
        columns={"segment": "reviewed_product_segment"}
    )
    promotions = pd.read_csv(data_path / "promotions.csv")
    reviews = pd.read_csv(data_path / "reviews.csv")

    orders_segment = orders.merge(customer_segments, on="customer_id", how="inner")
    items_segment = (
        order_items.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(customer_segments, on="customer_id", how="inner")
        .merge(product_lookup, on="product_id", how="left")
    )

    promo_long = (
        order_items[["order_id", "promo_id", "promo_id_2"]]
        .melt(id_vars="order_id", value_name="promo_code")
        .dropna(subset=["promo_code"])
        .drop(columns="variable")
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(customer_segments, on="customer_id", how="inner")
        .merge(
            promotions[["promo_id", "promo_name", "promo_type", "promo_channel"]],
            left_on="promo_code",
            right_on="promo_id",
            how="left",
        )
    )

    reviews_segment = (
        reviews.merge(customer_segments, on="customer_id", how="inner")
        .merge(review_product_lookup, on="product_id", how="left")
    )

    sources = {
        "Product category by units": (items_segment, "category", "quantity"),
        "Product segment by units": (items_segment, "product_segment", "quantity"),
        "Top products by units": (items_segment, "product_name", "quantity"),
        "City by customers": (customer_table, "city", None),
        "Acquisition channel": (customer_table, "acquisition_channel", None),
        "Dominant promo type": (customer_table, "dominant_promo_type", None),
        "Promo campaign touches": (promo_long, "promo_name", None),
        "Promo channel touches": (promo_long, "promo_channel", None),
        "Payment method by orders": (orders_segment, "payment_method", None),
        "Device type by orders": (orders_segment, "device_type", None),
        "Order source by orders": (orders_segment, "order_source", None),
        "Reviewed category": (reviews_segment, "category", None),
        "Reviewed product segment": (reviews_segment, "reviewed_product_segment", None),
        "Reviewed products": (reviews_segment, "product_name", None),
    }

    share_matrices: Dict[str, pd.DataFrame] = {}
    top_tables: Dict[str, pd.DataFrame] = {}
    for name, (source_df, category_col, weight_col) in sources.items():
        if category_col not in source_df.columns:
            continue
        matrix, top_table = _share_matrix(
            source_df,
            segment_col=segment_col,
            category_col=category_col,
            segment_order=segment_order,
            top_n=top_n,
            weight_col=weight_col,
        )
        if not matrix.empty:
            share_matrices[name] = matrix
            top_tables[name] = top_table

    return SegmentBehaviorProfiles(share_matrices=share_matrices, top_tables=top_tables)


def plot_segment_behavior_heatmaps(
    share_matrices: Mapping[str, pd.DataFrame],
    chart_names: Optional[Sequence[str]] = None,
    ncols: int = 2,
    figsize_per_chart: Tuple[int, int] = (8, 4),
):
    """Plot row-normalized segment behavior heatmaps."""
    import matplotlib.pyplot as plt

    selected_names = list(chart_names) if chart_names else list(share_matrices.keys())
    selected_names = [name for name in selected_names if name in share_matrices]
    if not selected_names:
        raise ValueError("No behavior matrices available to plot.")

    nrows = int(np.ceil(len(selected_names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_chart[0] * ncols, figsize_per_chart[1] * nrows),
        squeeze=False,
    )

    for ax, name in zip(axes.flat, selected_names):
        matrix = share_matrices[name]
        image = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=max(0.01, matrix.values.max()))
        ax.set_title(name)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=9)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix.iloc[row_idx, col_idx]
                if value >= 0.01:
                    ax.text(col_idx, row_idx, f"{value:.0%}", ha="center", va="center", fontsize=7)

        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    for ax in axes.flat[len(selected_names) :]:
        ax.axis("off")

    plt.tight_layout()
    return fig
