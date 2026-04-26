"""Customer distribution visualization on Vietnam map.

The module follows the spirit of the provided Visualize.py example, but is
adapted for the Datathon customer table. It can create an interactive Folium
map when Folium is installed, and otherwise falls back to a static Matplotlib
map so the notebook can still run in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
import struct
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "profit": 0.28,
    "frequency": 0.16,
    "retention": 0.15,
    "units": 0.12,
    "profit_margin": 0.10,
    "lifespan": 0.08,
    "rating": 0.05,
    "review_engagement": 0.03,
    "promo": -0.07,
    "refund": -0.03,
    "refund_rate": -0.04,
    "return_qty": -0.02,
}


# Approximate city-center coordinates for all cities in data/geography.csv.
# Coordinates are sufficient for city-level EDA visualization, not routing.
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Bac Giang": (21.2731, 106.1946),
    "Bac Lieu": (9.2940, 105.7278),
    "Bac Ninh": (21.1861, 106.0763),
    "Ben Tre": (10.2433, 106.3756),
    "Bien Hoa": (10.9574, 106.8427),
    "Buon Ma Thuot": (12.6667, 108.0500),
    "Ca Mau": (9.1768, 105.1524),
    "Cam Pha": (21.0167, 107.3000),
    "Can Tho": (10.0452, 105.7469),
    "Da Lat": (11.9404, 108.4583),
    "Da Nang": (16.0544, 108.2022),
    "Dong Hoi": (17.4689, 106.6223),
    "Ha Long": (20.9712, 107.0448),
    "Hai Phong": (20.8449, 106.6881),
    "Hanoi": (21.0278, 105.8342),
    "Ho Chi Minh City": (10.8231, 106.6297),
    "Hoi An": (15.8801, 108.3380),
    "Hue": (16.4637, 107.5909),
    "Kon Tum": (14.3497, 108.0005),
    "Lao Cai": (22.4809, 103.9755),
    "Long Xuyen": (10.3864, 105.4352),
    "My Tho": (10.3600, 106.3600),
    "Nam Dinh": (20.4388, 106.1621),
    "Nha Trang": (12.2388, 109.1967),
    "Ninh Binh": (20.2506, 105.9745),
    "Phan Rang-Thap Cham": (11.5643, 108.9886),
    "Phan Thiet": (10.9805, 108.2615),
    "Phu Ly": (20.5411, 105.9139),
    "Pleiku": (13.9833, 108.0000),
    "Quang Ngai": (15.1205, 108.7923),
    "Quy Nhon": (13.7820, 109.2190),
    "Rach Gia": (10.0125, 105.0809),
    "Soc Trang": (9.6025, 105.9739),
    "Son Tay": (21.1405, 105.5069),
    "Tam Ky": (15.5736, 108.4740),
    "Thai Nguyen": (21.5942, 105.8482),
    "Tra Vinh": (9.9513, 106.3346),
    "Tuy Hoa": (13.0955, 109.3209),
    "Uong Bi": (21.0343, 106.7705),
    "Viet Tri": (21.3227, 105.4020),
    "Vinh Long": (10.2537, 105.9722),
    "Vung Tau": (10.4114, 107.1362),
}


@dataclass(frozen=True)
class MapPaths:
    customer_table: Path
    boundary_shapefile: Path


def default_map_paths(project_root: Optional[Path] = None) -> MapPaths:
    """Return default customer table and Vietnam boundary paths."""
    root = project_root or Path.cwd()
    if not (root / "data").exists() and (root.parent / "data").exists():
        root = root.parent

    return MapPaths(
        customer_table=root / "data" / "truong.le" / "customer_golden_table.csv",
        boundary_shapefile=(
            root
            / "data"
            / "truong.le"
            / "vn_boundary"
            / "extracted_files"
            / "vietnam_Vietnam_Country_Boundary.shp"
        ),
    )


def robust_minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale values to 0-1 using 1st and 99th percentiles to reduce outliers."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())
    lo = values.quantile(0.01)
    hi = values.quantile(0.99)

    if math.isclose(float(hi), float(lo)):
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = ((values.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)

    return scaled if higher_is_better else 1 - scaled


def build_customer_score(
    customer_df: pd.DataFrame,
    feature_weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Create weighted customer score from value, engagement, cost, and risk.

    Promotion usage is included as a negative component because it represents
    discount dependency and marketing cost. `purchase_lifespan_days` and
    `review_count` are added with small positive weights because the current
    golden table already exposes them and they capture relationship depth and
    engagement, but they should not dominate profit, frequency, or retention.
    `refund_rate` complements total refund by measuring value leakage intensity
    independent of customer size.
    """
    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if feature_weights:
        weights.update(feature_weights)

    scored = customer_df.copy()
    components = {
        "profit": robust_minmax(scored["monetary_profit"], True),
        "frequency": robust_minmax(scored["frequency"], True),
        "retention": robust_minmax(scored["recency_days"], False),
        "units": robust_minmax(scored["total_units"], True),
        "profit_margin": robust_minmax(scored["profit_margin"], True),
        "lifespan": robust_minmax(
            scored.get("purchase_lifespan_days", pd.Series(0, index=scored.index)),
            True,
        ),
        "rating": robust_minmax(scored.get("avg_rating", pd.Series(0, index=scored.index)), True),
        "review_engagement": robust_minmax(
            scored.get("review_count", pd.Series(0, index=scored.index)),
            True,
        ),
        "promo": robust_minmax(scored.get("promo_usage_rate", pd.Series(0, index=scored.index)), True),
        "refund": robust_minmax(scored.get("total_refund", pd.Series(0, index=scored.index)), True),
        "refund_rate": robust_minmax(scored.get("refund_rate", pd.Series(0, index=scored.index)), True),
        "return_qty": robust_minmax(scored.get("total_return_qty", pd.Series(0, index=scored.index)), True),
    }

    raw_score = pd.Series(0.0, index=scored.index)
    max_possible = 0.0
    min_possible = 0.0
    for name, weight in weights.items():
        if name not in components:
            continue
        raw_score += components[name] * weight
        if weight >= 0:
            max_possible += weight
        else:
            min_possible += weight

    denominator = max_possible - min_possible
    scored["customer_score"] = ((raw_score - min_possible) / denominator).clip(0, 1) if denominator else 0.5
    for name, values in components.items():
        scored[f"{name}_score_component"] = values

    return scored


def _mode_or_unknown(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "Unknown"
    return str(non_null.mode().iloc[0])


def build_city_nodes(
    customer_df: pd.DataFrame,
    feature_weights: Optional[Mapping[str, float]] = None,
    city_coords: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Aggregate customer scores into map-ready city nodes."""
    coords = dict(CITY_COORDS)
    if city_coords:
        coords.update(city_coords)

    scored = build_customer_score(customer_df, feature_weights)
    scored["latitude"] = scored["city"].map(lambda city: coords.get(city, (np.nan, np.nan))[0])
    scored["longitude"] = scored["city"].map(lambda city: coords.get(city, (np.nan, np.nan))[1])
    scored = scored.dropna(subset=["latitude", "longitude"]).copy()

    top_customer = (
        scored.sort_values(["city", "customer_score"], ascending=[True, False])
        .groupby("city")
        .head(1)[["city", "customer_id", "gender", "age_group", "customer_score"]]
        .rename(
            columns={
                "customer_id": "top_customer_id",
                "gender": "top_customer_gender",
                "age_group": "top_customer_age_group",
                "customer_score": "top_customer_score",
            }
        )
    )

    city_nodes = (
        scored.groupby("city", as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            customers=("customer_id", "count"),
            score=("customer_score", "mean"),
            total_profit=("monetary_profit", "sum"),
            avg_profit=("monetary_profit", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_retention_score=("retention_score_component", "mean"),
            avg_units=("total_units", "mean"),
            avg_promo_usage=("promo_usage_rate", "mean"),
            avg_refund=("total_refund", "mean"),
            avg_refund_rate=("refund_rate", "mean"),
            avg_return_qty=("total_return_qty", "mean"),
            avg_lifespan=("purchase_lifespan_days", "mean"),
            avg_review_count=("review_count", "mean"),
            dominant_gender=("gender", _mode_or_unknown),
            dominant_age_group=("age_group", _mode_or_unknown),
            dominant_category=("preferred_category", _mode_or_unknown),
        )
        .merge(top_customer, on="city", how="left")
        .sort_values("score", ascending=False)
    )

    return city_nodes


def read_shapefile_polygons(shapefile_path: Path) -> List[List[Tuple[float, float]]]:
    """Minimal polygon reader for ESRI .shp files.

    Returns a list of rings as (longitude, latitude) tuples. This avoids a hard
    dependency on fiona/geopandas for a simple static boundary plot.
    """
    rings: List[List[Tuple[float, float]]] = []
    with Path(shapefile_path).open("rb") as shp:
        shp.seek(100)
        while True:
            header = shp.read(8)
            if len(header) < 8:
                break

            _, content_length_words = struct.unpack(">2i", header)
            content = shp.read(content_length_words * 2)
            if len(content) < 44:
                continue

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15, 25):
                continue

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + num_parts * 4
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_offset:points_offset]))
            parts.append(num_points)

            points = [
                struct.unpack("<2d", content[points_offset + idx * 16 : points_offset + (idx + 1) * 16])
                for idx in range(num_points)
            ]
            for start, end in zip(parts[:-1], parts[1:]):
                rings.append(points[start:end])

    return rings


def _popup_html(row: pd.Series) -> str:
    return f"""
    <b>{html.escape(str(row['city']))}</b><br>
    Score: {row['score']:.3f}<br>
    Customers: {int(row['customers']):,}<br>
    Total profit: {row['total_profit']:,.0f}<br>
    Avg frequency: {row['avg_frequency']:.2f}<br>
    Avg units: {row['avg_units']:.2f}<br>
    Avg promo usage: {row['avg_promo_usage']:.1%}<br>
    Avg refund: {row['avg_refund']:,.0f}<br>
    Avg refund rate: {row['avg_refund_rate']:.1%}<br>
    Avg return qty: {row['avg_return_qty']:.2f}<br>
    Avg lifespan days: {row['avg_lifespan']:.0f}<br>
    Avg review count: {row['avg_review_count']:.2f}<br>
    Dominant gender: {html.escape(str(row['dominant_gender']))}<br>
    Dominant age: {html.escape(str(row['dominant_age_group']))}<br>
    Dominant category: {html.escape(str(row['dominant_category']))}<br>
    Top customer: {html.escape(str(row['top_customer_id']))}<br>
    Top customer gender: {html.escape(str(row['top_customer_gender']))}<br>
    Top customer age: {html.escape(str(row['top_customer_age_group']))}
    """


def create_customer_distribution_map(
    customer_df: pd.DataFrame,
    boundary_shapefile: Optional[Path] = None,
    feature_weights: Optional[Mapping[str, float]] = None,
    prefer_interactive: bool = True,
):
    """Create a Vietnam customer distribution map.

    Color encodes the weighted score; marker radius encodes city customer count.
    Returns a Folium map when Folium is available, otherwise a Matplotlib figure.
    """
    city_nodes = build_city_nodes(customer_df, feature_weights)

    if prefer_interactive:
        try:
            return _create_folium_map(city_nodes, boundary_shapefile)
        except ImportError:
            pass

    return create_static_customer_distribution_map(city_nodes, boundary_shapefile)


def _create_folium_map(city_nodes: pd.DataFrame, boundary_shapefile: Optional[Path] = None):
    import folium
    from branca.colormap import LinearColormap

    score_min = float(city_nodes["score"].min())
    score_max = float(city_nodes["score"].max())
    colormap = LinearColormap(
        ["#edf8fb", "#b2e2e2", "#66c2a4", "#238b45", "#00441b"],
        vmin=score_min,
        vmax=score_max,
        caption="Weighted customer score",
    )

    gmap = folium.Map(location=[16.0, 107.8], zoom_start=6, tiles="cartodbpositron")
    colormap.add_to(gmap)

    if boundary_shapefile and Path(boundary_shapefile).exists():
        for ring in read_shapefile_polygons(Path(boundary_shapefile)):
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in ring],
                color="#4a5568",
                weight=1,
                opacity=0.55,
            ).add_to(gmap)

    max_customers = city_nodes["customers"].max()
    for _, row in city_nodes.iterrows():
        radius = 5 + 18 * math.sqrt(row["customers"] / max_customers)
        color = colormap(row["score"])
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.78,
            weight=1.2,
            popup=folium.Popup(_popup_html(row), max_width=330),
            tooltip=f"{row['city']}: score {row['score']:.3f}",
        ).add_to(gmap)

    return gmap


def create_static_customer_distribution_map(
    city_nodes: pd.DataFrame,
    boundary_shapefile: Optional[Path] = None,
):
    """Create a static Matplotlib fallback map."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 12))

    if boundary_shapefile and Path(boundary_shapefile).exists():
        for ring in read_shapefile_polygons(Path(boundary_shapefile)):
            if not ring:
                continue
            lon, lat = zip(*ring)
            ax.plot(lon, lat, color="#64748b", linewidth=0.7, alpha=0.7)

    sizes = 40 + 520 * np.sqrt(city_nodes["customers"] / city_nodes["customers"].max())
    scatter = ax.scatter(
        city_nodes["longitude"],
        city_nodes["latitude"],
        c=city_nodes["score"],
        s=sizes,
        cmap="YlGn",
        edgecolor="#1f2937",
        linewidth=0.7,
        alpha=0.88,
    )

    top_labels = city_nodes.nlargest(10, "score")
    for _, row in top_labels.iterrows():
        ax.annotate(
            row["city"],
            (row["longitude"], row["latitude"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Weighted customer score")
    ax.set_title("Vietnam Customer Distribution by Weighted Score")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(102, 110.5)
    ax.set_ylim(8, 23.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def load_customer_distribution_inputs(
    customer_table_path: Optional[Path] = None,
    boundary_shapefile_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path]:
    """Load customer golden table and resolve the Vietnam boundary path."""
    paths = default_map_paths(project_root)
    table_path = Path(customer_table_path) if customer_table_path else paths.customer_table
    boundary_path = Path(boundary_shapefile_path) if boundary_shapefile_path else paths.boundary_shapefile
    return pd.read_csv(table_path), boundary_path
