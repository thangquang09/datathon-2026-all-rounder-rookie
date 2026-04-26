"""Objective-aware RFM segmentation utilities.

These helpers keep the notebook's rule-based RFM segmentation explainable,
then add a data-calibrated objective score and KMeans diagnostic layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SEGMENT_LABELS: List[str] = ["Champions", "Loyal", "Potential", "At Risk", "Lost"]
SEGMENT_ORDER: List[str] = ["Champions", "Loyal", "Potential", "Need Attention", "At Risk", "Lost"]
FINAL_SEGMENT_ORDER: List[str] = SEGMENT_ORDER.copy()
SEGMENT_TIER: Dict[str, int] = {segment: tier for tier, segment in enumerate(SEGMENT_ORDER, start=1)}
TIER_SEGMENT: Dict[int, str] = {tier: segment for segment, tier in SEGMENT_TIER.items()}


DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "profit": 0.32,
    "frequency": 0.18,
    "retention": 0.18,
    "margin": 0.10,
    "lifespan": 0.07,
    "satisfaction": 0.03,
    "review_engagement": 0.02,
    "promo": -0.08,
    "refund": -0.08,
    "refund_rate": -0.05,
    "return_qty": -0.04,
}


DEFAULT_CLUSTER_FEATURES: List[str] = [
    "recency_days",
    "frequency",
    "monetary_profit",
    "avg_order_value",
    "promo_usage_rate",
    "profit_margin",
    "total_units",
    "refund_rate",
    "total_return_qty",
    "purchase_lifespan_days",
    "avg_rating",
    "review_count",
    "objective_score",
]


@dataclass(frozen=True)
class ObjectiveSegmentationResult:
    """Container returned by the full objective-aware segmentation workflow."""

    customer_table: pd.DataFrame
    segment_summary: pd.DataFrame
    score_segment_summary: pd.DataFrame
    final_segment_summary: pd.DataFrame
    cluster_profile: pd.DataFrame
    cluster_segment_distribution: pd.DataFrame
    cluster_explanations: pd.DataFrame
    silhouette: Optional[float]


def robust_minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale one feature to 0-1 using data-driven robust bounds.

    The current customer table has very skewed commercial variables: profit,
    revenue, frequency, total units, refunds, and return quantity all have long
    upper tails. A plain min-max transform would let a tiny number of extreme
    customers dominate the score. This function therefore uses the 1st and 99th
    percentiles from the actual dataset as comparison numbers. Values outside
    that empirical range are clipped, which keeps the scoring sensitive to the
    bulk of customers while still rewarding high performers.

    `higher_is_better=False` is used for risk/cost features such as recency if
    they are represented directly, because fewer days since last order is better
    than more days.
    """
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = values.median()
    values = values.fillna(median if pd.notna(median) else 0)
    low = values.quantile(0.01)
    high = values.quantile(0.99)

    if np.isclose(high, low):
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = ((values.clip(low, high) - low) / (high - low)).clip(0, 1)

    return scaled if higher_is_better else 1 - scaled


def add_rfm_scores(customer_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Add R, F, M quintile scores based on this dataset's rank distribution.

    The comparison numbers in the later RFM rules are based on these quintiles,
    not on a generic internet benchmark. With this dataset, customer value is
    highly concentrated and frequency/recency are skewed, so using ranks before
    `qcut` gives equally sized score bands and avoids ties around common values
    like one-time or four-time buyers. A score of 4 or 5 therefore means "top
    40% within this customer base"; a score of 1 or 2 means "bottom 40% within
    this customer base".

    Monetary is deliberately `monetary_profit` rather than revenue because the
    business objective in this notebook is profit and LTV optimization.
    """
    scored = customer_df.copy()
    labels = list(range(1, n_bins + 1))

    scored["R_score"] = pd.qcut(
        scored["recency_days"].rank(method="first", ascending=False),
        n_bins,
        labels=labels,
    ).astype(int)
    scored["F_score"] = pd.qcut(
        scored["frequency"].rank(method="first"),
        n_bins,
        labels=labels,
    ).astype(int)
    scored["M_score"] = pd.qcut(
        scored["monetary_profit"].rank(method="first"),
        n_bins,
        labels=labels,
    ).astype(int)
    scored["RFM_score"] = (
        scored["R_score"].astype(str)
        + scored["F_score"].astype(str)
        + scored["M_score"].astype(str)
    )

    return scored


def assign_rfm_segments(
    customer_df: pd.DataFrame,
    output_col: str = "segment",
    segment_labels: Sequence[str] = SEGMENT_LABELS,
) -> pd.DataFrame:
    """Assign explainable RFM business labels from dataset-relative scores.

    These rules are intentionally simple because their role is storytelling and
    operational interpretation. The thresholds are chosen from the quintile
    score bands created from the current data:

    - `>= 4` means the customer is in the top 40% of this customer base for that
      R/F/M dimension, which is broad enough to produce stable business groups
      while still separating high-value customers from the long tail.
    - `<= 2` means the customer is in the bottom 40%, which captures dormant or
      thin-history customers in this dataset where recency has a long tail.
    - `F_score >= 3` for `At Risk` means the customer has at least median-level
      purchase history, so they are not just inactive; they are customers the
      business previously had enough relationship with to win back.

    The labels are business interpretations of the observed score pattern:
    `Champions` are top-tier on recency, frequency, and profit; `Loyal` are
    frequent and profitable even if not very recent; `Potential` are still warm
    because recency is high; `At Risk` have relationship depth but weak recency;
    `Lost` are weak on both recency and frequency. `Need Attention` is kept as
    a middle/default group for mixed cases that the hard rules should not force
    into a more confident label.
    """
    if len(segment_labels) != 5:
        raise ValueError("segment_labels must contain exactly five labels.")

    segmented = customer_df.copy()
    champion, loyal, potential, at_risk, lost = segment_labels
    conditions = [
        (segmented["R_score"] >= 4) & (segmented["F_score"] >= 4) & (segmented["M_score"] >= 4),
        (segmented["F_score"] >= 4) & (segmented["M_score"] >= 3),
        (segmented["R_score"] >= 4) & ((segmented["F_score"] >= 2) | (segmented["M_score"] >= 2)),
        (segmented["R_score"] <= 2) & (segmented["F_score"] >= 3),
        (segmented["R_score"] <= 2) & (segmented["F_score"] <= 2),
    ]
    segmented[output_col] = np.select(conditions, [champion, loyal, potential, at_risk, lost], default="Need Attention")
    return segmented


def add_objective_score(
    customer_df: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
    output_col: str = "objective_score",
) -> pd.DataFrame:
    """Add a weighted objective score calibrated for this customer table.

    The default weights are intentionally not equal. They reflect what the
    current EDA shows about this dataset:

    - Profit gets the largest weight (`0.32`) because profit is highly
      concentrated among a small share of customers; optimizing revenue alone
      would overvalue low-margin or promotion-heavy buyers.
    - Frequency and retention each get `0.18` because the dataset contains many
      one-time or dormant customers, so repeat purchase and recent activity are
      strong signals of reachable LTV.
    - Margin gets `0.10` because two customers with similar profit can have very
      different economics if one requires heavy discounting.
    - Purchase lifespan gets `0.07` because it separates customers with a real
      relationship history from customers who only bought once recently.
    - Satisfaction and review engagement get small positive weights (`0.03` and
      `0.02`) because `avg_rating` and `review_count` are available but should
      not outweigh observed commercial behavior.
    - Promotion usage is a negative weight (`-0.08`) because the current table
      has a high median promo usage rate, and promotion-heavy customers can hide
      acquisition or discount cost even when revenue looks healthy.
    - Refund amount, refund rate, and return quantity are negative weights
      (`-0.08`, `-0.05`, and `-0.04`) as value leakage penalties. Amount captures
      absolute leakage; rate captures leakage intensity independent of customer
      size; return quantity captures operational friction.

    The final score is rescaled to 0-1 after combining positive value features
    and negative risk features. If the objective changes, pass another `weights`
    dictionary, for example increasing `retention` for retention campaigns or
    making `promo` more negative for marketing-efficiency analysis.
    """
    score_df = customer_df.copy()
    final_weights = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if weights:
        final_weights.update(weights)

    components = {
        "profit": robust_minmax_score(score_df["monetary_profit"], True),
        "frequency": robust_minmax_score(score_df["frequency"], True),
        "retention": robust_minmax_score(score_df["recency_days"], False),
        "margin": robust_minmax_score(score_df["profit_margin"], True),
        "lifespan": robust_minmax_score(
            score_df.get("purchase_lifespan_days", pd.Series(0, index=score_df.index)),
            True,
        ),
        "satisfaction": robust_minmax_score(
            score_df.get("avg_rating", pd.Series(0, index=score_df.index)),
            True,
        ),
        "review_engagement": robust_minmax_score(
            score_df.get("review_count", pd.Series(0, index=score_df.index)),
            True,
        ),
        "promo": robust_minmax_score(
            score_df.get("promo_usage_rate", pd.Series(0, index=score_df.index)),
            True,
        ),
        "refund": robust_minmax_score(score_df["total_refund"], True),
        "refund_rate": robust_minmax_score(
            score_df.get("refund_rate", pd.Series(0, index=score_df.index)),
            True,
        ),
        "return_qty": robust_minmax_score(score_df["total_return_qty"], True),
    }

    raw_score = pd.Series(0.0, index=score_df.index)
    min_possible = 0.0
    max_possible = 0.0
    for name, weight in final_weights.items():
        if name not in components:
            continue
        raw_score += components[name] * weight
        if weight >= 0:
            max_possible += weight
        else:
            min_possible += weight
        score_df[f"{name}_score"] = components[name]

    denominator = max_possible - min_possible
    score_df[output_col] = ((raw_score - min_possible) / denominator).clip(0, 1) if denominator else 0.5
    return score_df


def assign_score_segments(
    customer_df: pd.DataFrame,
    score_col: str = "objective_score",
    reference_col: str = "segment",
    output_col: str = "score_segment",
    tier_col: str = "score_tier",
    segment_order: Sequence[str] = SEGMENT_ORDER,
) -> pd.DataFrame:
    """Create objective-score tiers using the RFM segment distribution.

    The score tier deliberately uses the *same customer counts* as the current
    rule-based RFM labels. This is data-specific and avoids creating a second
    segmentation system with arbitrary equal-sized quintiles. If the RFM rules
    classify 20,598 customers as `Champions`, then the top 20,598 customers by
    `objective_score` are assigned score tier 1 / `Champions`; the next block
    receives tier 2 / `Loyal`, and so on through `Lost`.

    This makes the comparison fair: RFM and score tiers use the same vocabulary
    and the same distribution, but they rank customers from different evidence.
    RFM captures explainable behavior patterns, while the weighted score
    captures the profit-retention-refund objective.
    """
    segmented = customer_df.copy()
    ordered_segments = [segment for segment in segment_order if segment in segmented[reference_col].unique()]
    counts = segmented[reference_col].value_counts().reindex(ordered_segments).fillna(0).astype(int)
    sorted_index = segmented.sort_values(score_col, ascending=False, kind="mergesort").index

    segmented[output_col] = pd.NA
    segmented[tier_col] = pd.NA

    start = 0
    for tier, segment in enumerate(ordered_segments, start=1):
        stop = start + int(counts.loc[segment])
        tier_index = sorted_index[start:stop]
        segmented.loc[tier_index, output_col] = segment
        segmented.loc[tier_index, tier_col] = tier
        start = stop

    if start < len(sorted_index):
        fallback_segment = ordered_segments[-1]
        fallback_tier = len(ordered_segments)
        segmented.loc[sorted_index[start:], output_col] = fallback_segment
        segmented.loc[sorted_index[start:], tier_col] = fallback_tier

    segmented[tier_col] = segmented[tier_col].astype(int)
    return segmented


def assign_final_segments(
    customer_df: pd.DataFrame,
    rfm_col: str = "segment",
    score_segment_col: str = "score_segment",
    rfm_tier_col: str = "rfm_tier",
    score_tier_col: str = "score_tier",
    final_tier_col: str = "final_tier",
    output_col: str = "final_segment",
    segment_order: Sequence[str] = SEGMENT_ORDER,
) -> pd.DataFrame:
    """Merge RFM and objective-score tiers into one six-label segmentation.

    The final segment keeps the original six business labels instead of
    expanding into many sub-labels. First, RFM assigns an explainable behavior
    tier: `Champions` = 1, `Loyal` = 2, ..., `Lost` = 6. Separately,
    `assign_score_segments` ranks `objective_score` into the same six labels and
    the same distribution as RFM. The final tier is then:

    `floor((rfm_tier + score_tier) / 2)`

    With this data, that rule is intentionally conservative toward the better
    tier because RFM already encodes recent/frequent/profitable behavior. For
    example, a `Champions` customer with score tier 2 remains tier 1
    (`floor(1.5)=1`), while a `Champions` customer with score tier 3 moves to
    tier 2 (`floor(2)=2`). This matches the business interpretation requested in
    the notebook: objective score can pull a customer up or down, but the final
    vocabulary remains simple enough for marketing action.
    """
    segmented = customer_df.copy()
    tier_lookup = {segment: tier for tier, segment in enumerate(segment_order, start=1)}
    label_lookup = {tier: segment for segment, tier in tier_lookup.items()}

    segmented[rfm_tier_col] = segmented[rfm_col].map(tier_lookup).astype(int)
    if score_tier_col not in segmented.columns:
        segmented[score_tier_col] = segmented[score_segment_col].map(tier_lookup).astype(int)

    averaged_tier = (segmented[rfm_tier_col] + segmented[score_tier_col]) / 2
    segmented[final_tier_col] = np.floor(averaged_tier).clip(1, len(segment_order)).astype(int)
    segmented[output_col] = segmented[final_tier_col].map(label_lookup)
    return segmented


def summarize_segments(
    customer_df: pd.DataFrame,
    segment_col: str = "segment",
    segment_order: Sequence[str] = SEGMENT_ORDER,
) -> pd.DataFrame:
    """Summarize RFM or score segments for business review."""
    summary = (
        customer_df.groupby(segment_col)
        .agg(
            customers=("customer_id", "count"),
            avg_profit=("monetary_profit", "mean"),
            total_profit=("monetary_profit", "sum"),
            avg_frequency=("frequency", "mean"),
            avg_recency=("recency_days", "mean"),
            avg_aov=("avg_order_value", "mean"),
            avg_margin=("profit_margin", "mean"),
            promo_usage=("promo_usage_rate", "mean"),
            avg_lifespan=("purchase_lifespan_days", "mean"),
            avg_rating=("avg_rating", "mean"),
            avg_review_count=("review_count", "mean"),
            refund_rate=("refund_rate", "mean"),
            avg_objective_score=("objective_score", "mean"),
        )
    )
    ordered = [label for label in segment_order if label in summary.index]
    remaining = [label for label in summary.index if label not in ordered]
    summary = summary.reindex(ordered + remaining)
    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    summary["profit_share"] = summary["total_profit"] / summary["total_profit"].sum()
    return summary


def prepare_kmeans_features(
    customer_df: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_CLUSTER_FEATURES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare KMeans features with transforms chosen for this dataset.

    KMeans is distance-based, while this customer table mixes days, money,
    rates, counts, and very skewed totals. The preparation is therefore based on
    the observed data shape:

    - Monetary and count features are clipped at zero and transformed with
      `log1p` because profit, frequency, AOV, units, and return quantities have
      long right tails.
    - `recency_days` is also log-transformed because inactivity has a long tail.
    - Ratio features such as promo usage, profit margin, refund rate, and the
      objective score stay on their natural scale before standardization.
    - Finally, `StandardScaler` makes each feature contribute by pattern rather
      than by unit size.
    """
    available_cols = [col for col in feature_cols if col in customer_df.columns]
    features = customer_df[available_cols].copy()
    features = features.replace([np.inf, -np.inf], np.nan)

    for col in available_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].fillna(features[col].median())

    log_cols = [
        col
        for col in [
            "recency_days",
            "frequency",
            "monetary_profit",
            "avg_order_value",
            "total_units",
            "total_return_qty",
        ]
        if col in features.columns
    ]
    for col in log_cols:
        features[col] = np.log1p(features[col].clip(lower=0))

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=available_cols,
        index=customer_df.index,
    )
    return features, scaled


def run_kmeans_segment_diagnostics(
    customer_df: pd.DataFrame,
    segment_col: str = "segment",
    n_clusters: Optional[int] = None,
    feature_cols: Sequence[str] = DEFAULT_CLUSTER_FEATURES,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[float]]:
    """Run KMeans and compare natural clusters with final business labels.

    `n_clusters` defaults to `len(SEGMENT_ORDER)` because the goal is not to
    claim this is the mathematically optimal K. The goal is diagnostic: compare
    six natural behavior groups against the final labels the business wants to
    reason about. Those labels combine RFM behavior with within-behavior
    objective priority. If a cluster is dominated by one final label, the rules
    are aligned with the data geometry. If a cluster mixes labels, the rules are
    probably grouping customers that behave differently once refunds, units,
    margin, promo usage, and objective score are considered.
    """
    k = n_clusters or len(SEGMENT_ORDER)
    diagnostics = customer_df.copy()
    _, scaled_features = prepare_kmeans_features(diagnostics, feature_cols)

    with warnings.catch_warnings():
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        warnings.filterwarnings("ignore", category=UserWarning)
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        diagnostics["kmeans_cluster"] = kmeans.fit_predict(scaled_features)

    silhouette = None
    if len(diagnostics) > k and len(set(diagnostics["kmeans_cluster"])) > 1:
        sample_size = min(3000, len(diagnostics))
        sample = scaled_features.sample(sample_size, random_state=random_state)
        silhouette = float(silhouette_score(sample, diagnostics.loc[sample.index, "kmeans_cluster"]))

    cluster_profile = (
        diagnostics.groupby("kmeans_cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_profit=("monetary_profit", "mean"),
            avg_margin=("profit_margin", "mean"),
            promo_usage=("promo_usage_rate", "mean"),
            refund_rate=("refund_rate", "mean"),
            return_qty=("total_return_qty", "mean"),
            avg_lifespan=("purchase_lifespan_days", "mean"),
            avg_rating=("avg_rating", "mean"),
            avg_review_count=("review_count", "mean"),
            avg_score=("objective_score", "mean"),
        )
        .sort_index()
    )
    cluster_profile["customer_share"] = cluster_profile["customers"] / cluster_profile["customers"].sum()

    cluster_segment_distribution = pd.crosstab(
        diagnostics["kmeans_cluster"],
        diagnostics[segment_col],
        normalize="index",
    ).reindex(index=cluster_profile.index).fillna(0)

    cluster_explanations = explain_cluster_segment_distribution(
        cluster_profile,
        cluster_segment_distribution,
    )

    return diagnostics, cluster_profile, cluster_segment_distribution, cluster_explanations, silhouette


def explain_cluster_segment_distribution(
    cluster_profile: pd.DataFrame,
    cluster_segment_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Explain why one final segment label dominates or stays small in each cluster.

    The explanation is generated from the cluster's own profile relative to the
    full customer table profile. For example, high frequency and high score make
    a `Champions` or `Loyal` dominance interpretable; high recency makes
    `At Risk` or `Lost` more natural; a small minority label usually appears
    because the hard RFM thresholds place it elsewhere even though its scaled
    behavior is close to the dominant cluster shape.
    """
    global_medians = cluster_profile.median(numeric_only=True)
    rows = []

    for cluster_id, distribution in cluster_segment_distribution.iterrows():
        sorted_distribution = distribution.sort_values(ascending=False)
        dominant_label = sorted_distribution.index[0]
        dominant_share = sorted_distribution.iloc[0]
        smallest_label = sorted_distribution.index[-1]
        smallest_share = sorted_distribution.iloc[-1]
        profile = cluster_profile.loc[cluster_id]

        traits = []
        if profile["avg_score"] >= global_medians["avg_score"]:
            traits.append("above-median objective score")
        else:
            traits.append("below-median objective score")
        if profile["avg_profit"] >= global_medians["avg_profit"]:
            traits.append("higher profit")
        else:
            traits.append("lower profit")
        if profile["avg_frequency"] >= global_medians["avg_frequency"]:
            traits.append("higher repeat purchase")
        else:
            traits.append("lower repeat purchase")
        if profile["avg_recency"] >= global_medians["avg_recency"]:
            traits.append("weaker recency")
        else:
            traits.append("stronger recency")
        if profile["refund_rate"] >= global_medians["refund_rate"]:
            traits.append("higher refund pressure")
        else:
            traits.append("lower refund pressure")

        explanation = (
            f"{dominant_label} dominates because this cluster shows "
            + ", ".join(traits)
            + ". "
            + f"{smallest_label} remains small because customers with that final label do not usually share this cluster's combined value, recency, frequency, and risk profile."
        )

        rows.append(
            {
                "kmeans_cluster": cluster_id,
                "dominant_segment": dominant_label,
                "dominant_share": dominant_share,
                "smallest_segment": smallest_label,
                "smallest_share": smallest_share,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(rows)


def plot_kmeans_segment_distribution(
    cluster_segment_distribution: pd.DataFrame,
    segment_order: Sequence[str] = FINAL_SEGMENT_ORDER,
    figsize: Tuple[int, int] = (12, 6),
):
    """Visualize the percentage mix of final business labels in each cluster.

    This chart is easier to read than a crosstab when the goal is diagnostic:
    a cluster with one dominant color means the RFM rule and KMeans geometry are
    aligned, while a highly mixed bar means that customers with different RFM
    behaviors or priorities look similar once objective score, refund behavior, margin, units,
    and promo usage are considered.
    """
    import matplotlib.pyplot as plt

    distribution = cluster_segment_distribution.copy()
    ordered_cols = [col for col in segment_order if col in distribution.columns]
    ordered_cols += [col for col in distribution.columns if col not in ordered_cols]
    distribution = distribution[ordered_cols]

    colors = {
        "Champions": "#0b6e4f",
        "Loyal": "#2a9d8f",
        "Potential": "#e9c46a",
        "Need Attention": "#f4a261",
        "At Risk": "#e76f51",
        "Lost": "#8d99ae",
    }

    def pick_color(label: str) -> str:
        if "Champion" in label:
            return "#0b6e4f"
        if "Loyal" in label:
            return "#2a9d8f"
        if "Potential" in label or "Nurture" in label:
            return "#e9c46a"
        if "Attention" in label:
            return "#f4a261"
        if "Risk" in label or "Win-Back" in label:
            return "#e76f51"
        if "Lost" in label or "Dormant" in label or "Automation" in label:
            return "#8d99ae"
        return "#64748b"

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(distribution))
    x = np.arange(len(distribution))

    for col in distribution.columns:
        values = distribution[col].values
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=col,
            color=colors.get(col, pick_color(str(col))),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {idx}" for idx in distribution.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of customers within cluster")
    ax.set_title("KMeans Cluster Composition by Final Segment")
    ax.legend(title="Final segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    plt.tight_layout()
    return fig


def plot_kmeans_segment_heatmap(
    cluster_segment_distribution: pd.DataFrame,
    segment_order: Sequence[str] = FINAL_SEGMENT_ORDER,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot cluster-vs-final-segment shares as a heatmap.

    The heatmap complements the stacked bar: it makes dominant labels and small
    minority labels visually explicit, which helps explain why a final label is
    dominant in one cluster but only a small percentage in another.
    """
    import matplotlib.pyplot as plt

    distribution = cluster_segment_distribution.copy()
    ordered_cols = [col for col in segment_order if col in distribution.columns]
    ordered_cols += [col for col in distribution.columns if col not in ordered_cols]
    distribution = distribution[ordered_cols]

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(distribution.values, cmap="YlGnBu", vmin=0, vmax=distribution.values.max())
    ax.set_xticks(np.arange(len(distribution.columns)))
    ax.set_xticklabels(distribution.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(distribution.index)))
    ax.set_yticklabels([f"Cluster {idx}" for idx in distribution.index])
    ax.set_title("Final Segment Share Inside Each KMeans Cluster")

    for row_idx in range(distribution.shape[0]):
        for col_idx in range(distribution.shape[1]):
            value = distribution.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.0%}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Share within cluster")
    plt.tight_layout()
    return fig


def plot_kmeans_cluster_profile(
    cluster_profile: pd.DataFrame,
    figsize: Tuple[int, int] = (11, 6),
):
    """Plot cluster profile using score, profit, frequency, and cluster size.

    The x-axis is average objective score, the y-axis is average profit, bubble
    size is cluster population, and color encodes average frequency. This gives
    a compact view of why a cluster tends to be dominated by `Champions`,
    `Loyal`, or `Lost` before reading the text explanation.
    """
    import matplotlib.pyplot as plt

    profile = cluster_profile.copy()
    fig, ax = plt.subplots(figsize=figsize)
    sizes = 180 + 1800 * profile["customers"] / profile["customers"].max()
    scatter = ax.scatter(
        profile["avg_score"],
        profile["avg_profit"],
        s=sizes,
        c=profile["avg_frequency"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="#1f2937",
        linewidth=0.8,
    )

    for cluster_id, row in profile.iterrows():
        ax.annotate(
            f"Cluster {cluster_id}",
            (row["avg_score"], row["avg_profit"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=10,
        )

    ax.set_title("KMeans Cluster Profile: Score, Profit, Frequency, Size")
    ax.set_xlabel("Average objective score")
    ax.set_ylabel("Average monetary profit")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Average frequency")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def apply_objective_rfm_workflow(
    customer_df: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
    segment_labels: Sequence[str] = SEGMENT_LABELS,
    n_clusters: Optional[int] = None,
) -> ObjectiveSegmentationResult:
    """Apply RFM rules, distribution-aligned score tiers, and KMeans diagnostics.

    This is the function the notebook imports in Section 5. It preserves the
    original rule-based RFM segmentation for explainability, adds a weighted
    objective score, maps that score into the same six-label distribution as
    RFM, merges both into one `final_segment`, then runs KMeans with six
    clusters to see whether the final business labels align with natural
    customer behavior in this dataset.
    """
    table = add_rfm_scores(customer_df)
    table = assign_rfm_segments(table, segment_labels=segment_labels, output_col="segment")
    table = add_objective_score(table, weights=weights, output_col="objective_score")
    table = assign_score_segments(
        table,
        score_col="objective_score",
        reference_col="segment",
        output_col="score_segment",
        tier_col="score_tier",
        segment_order=SEGMENT_ORDER,
    )
    table = assign_final_segments(
        table,
        rfm_col="segment",
        score_segment_col="score_segment",
        rfm_tier_col="rfm_tier",
        score_tier_col="score_tier",
        final_tier_col="final_tier",
        output_col="final_segment",
        segment_order=SEGMENT_ORDER,
    )

    segment_summary = summarize_segments(table, segment_col="segment")
    score_segment_summary = summarize_segments(
        table,
        segment_col="score_segment",
        segment_order=SEGMENT_ORDER,
    )
    final_segment_summary = summarize_segments(
        table,
        segment_col="final_segment",
        segment_order=FINAL_SEGMENT_ORDER,
    )

    (
        table_with_clusters,
        cluster_profile,
        cluster_segment_distribution,
        cluster_explanations,
        silhouette,
    ) = run_kmeans_segment_diagnostics(
        table,
        segment_col="final_segment",
        n_clusters=n_clusters or len(SEGMENT_ORDER),
    )

    return ObjectiveSegmentationResult(
        customer_table=table_with_clusters,
        segment_summary=segment_summary,
        score_segment_summary=score_segment_summary,
        final_segment_summary=final_segment_summary,
        cluster_profile=cluster_profile,
        cluster_segment_distribution=cluster_segment_distribution,
        cluster_explanations=cluster_explanations,
        silhouette=silhouette,
    )
