from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = ROOT / "data" / "sales.csv"
HOLIDAY_PATH = ROOT / "docs" / "vietnam_holiday_calendar_2012_2024.csv"
OUT_DIR = ROOT / "model_thang" / "artifacts" / "holiday_impact"

EVENTS = {
    "New Year": "new_year",
    "Tet Eve": "tet_eve",
    "Tet Nguyen Dan": "tet_nguyen_dan",
    "Hung Kings": "hung_kings_10_3_lunar",
    "Liberation Day": "liberation_day",
    "Labour Day": "labour_day",
    "National Day": "national_day",
    "Mid-Autumn": "mid_autumn_15_8_lunar",
    "Black Friday": "black_friday",
    "Cyber Monday": "cyber_monday",
    "Valentine": "valentine",
    "Women Mar 8": "womens_day_mar8",
    "Women Oct 20": "womens_day_oct20",
    "Teachers Day": "teachers_day",
    "Christmas": "christmas",
}
DRILLDOWN_EVENTS = ["Tet Nguyen Dan", "Hung Kings"]
TARGETS = ["Revenue", "COGS"]
WINDOW = 30
BASELINE_INNER = 31
BASELINE_OUTER = 60


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(SALES_PATH, parse_dates=["Date"]).sort_values("Date")
    holidays = pd.read_csv(HOLIDAY_PATH)
    for col in holidays.columns:
        if col != "year":
            holidays[col] = pd.to_datetime(holidays[col])
    return sales, holidays


def local_baseline(sales: pd.DataFrame, event_date: pd.Timestamp, target: str) -> float:
    rel_day = (sales["Date"] - event_date).dt.days
    mask = rel_day.abs().between(BASELINE_INNER, BASELINE_OUTER)
    baseline = sales.loc[mask, target].median()
    if not np.isfinite(baseline) or baseline <= 0:
        year_mask = sales["Date"].dt.year.eq(event_date.year)
        baseline = sales.loc[year_mask, target].median()
    return float(baseline)


def build_event_panel(sales: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    date_min, date_max = sales["Date"].min(), sales["Date"].max()

    for event_name, col in EVENTS.items():
        for _, holiday in holidays.iterrows():
            event_date = holiday[col]
            if pd.isna(event_date):
                continue
            start = event_date - pd.Timedelta(days=WINDOW)
            end = event_date + pd.Timedelta(days=WINDOW)
            if start < date_min or end > date_max:
                continue

            segment = sales.loc[sales["Date"].between(start, end), ["Date", *TARGETS]].copy()
            if segment.empty:
                continue
            segment["event"] = event_name
            segment["year"] = int(holiday["year"])
            segment["event_date"] = event_date
            segment["rel_day"] = (segment["Date"] - event_date).dt.days

            for target in TARGETS:
                baseline = local_baseline(sales, event_date, target)
                segment[f"{target}_baseline"] = baseline
                segment[f"{target}_pct_vs_baseline"] = segment[target] / baseline - 1.0
                segment[f"{target}_delta_vs_baseline"] = segment[target] - baseline
            rows.append(segment)

    if not rows:
        raise RuntimeError("No full event windows found in sales history.")
    return pd.concat(rows, ignore_index=True)


def summarize(panel: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (event, year), g in panel.groupby(["event", "year"], sort=True):
        for target in TARGETS:
            day0 = g.loc[g["rel_day"].eq(0), f"{target}_pct_vs_baseline"].mean()
            before = g.loc[g["rel_day"].between(-7, -1), f"{target}_pct_vs_baseline"].mean()
            during = g.loc[g["rel_day"].between(-3, 3), f"{target}_pct_vs_baseline"].mean()
            after = g.loc[g["rel_day"].between(1, 7), f"{target}_pct_vs_baseline"].mean()
            summary_rows.append(
                {
                    "event": event,
                    "year": year,
                    "target": target,
                    "day0_pct_vs_local_baseline": day0,
                    "pre_7d_pct_vs_local_baseline": before,
                    "event_7d_pct_vs_local_baseline": during,
                    "post_7d_pct_vs_local_baseline": after,
                }
            )
    return pd.DataFrame(summary_rows)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        summary.groupby(["event", "target"], as_index=False)
        .agg(
            day0_mean=("day0_pct_vs_local_baseline", "mean"),
            pre7_mean=("pre_7d_pct_vs_local_baseline", "mean"),
            event7_mean=("event_7d_pct_vs_local_baseline", "mean"),
            post7_mean=("post_7d_pct_vs_local_baseline", "mean"),
            event7_std=("event_7d_pct_vs_local_baseline", "std"),
            n_years=("year", "nunique"),
        )
    )
    aggregate["abs_event7_mean"] = aggregate["event7_mean"].abs()
    aggregate["rank_score"] = aggregate.groupby("event")["abs_event7_mean"].transform("mean")
    return aggregate.sort_values(["rank_score", "event", "target"], ascending=[False, True, True])


def plot_event_study(panel: pd.DataFrame, events: list[str], filename: str, title: str) -> Path:
    long = panel.melt(
        id_vars=["event", "year", "rel_day"],
        value_vars=[f"{target}_pct_vs_baseline" for target in TARGETS],
        var_name="target",
        value_name="pct_vs_baseline",
    )
    long["target"] = long["target"].str.replace("_pct_vs_baseline", "", regex=False)
    long = long[long["event"].isin(events)]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(len(events), 2, figsize=(18, 4.8 * len(events)), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    for ax, (event, target) in zip(axes.ravel(), [(e, t) for e in events for t in TARGETS]):
        g = long[(long["event"] == event) & (long["target"] == target)]
        sns.lineplot(
            data=g,
            x="rel_day",
            y="pct_vs_baseline",
            estimator="mean",
            errorbar=("pi", 50),
            color="#1f77b4" if target == "Revenue" else "#d62728",
            ax=ax,
        )
        ax.axhline(0, color="#333333", linewidth=1)
        ax.axvline(0, color="#111111", linewidth=1.5, linestyle="--")
        ax.axvspan(-3, 3, color="#ffbf69", alpha=0.18)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
        ax.set_title(f"{event} impact on {target}")
        ax.set_xlabel("Days from event")
        ax.set_ylabel("% vs local baseline")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmaps(panel: pd.DataFrame, events: list[str], filename: str, title: str) -> Path:
    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(len(events), 2, figsize=(18, 4.3 * len(events)), sharex=True)
    axes = np.atleast_2d(axes)

    for ax, (event, target) in zip(axes.ravel(), [(e, t) for e in events for t in TARGETS]):
        g = panel[panel["event"].eq(event)]
        pivot = g.pivot_table(
            index="year",
            columns="rel_day",
            values=f"{target}_pct_vs_baseline",
            aggfunc="mean",
        ).sort_index()
        sns.heatmap(
            pivot,
            cmap="RdBu_r",
            center=0,
            vmin=-0.7,
            vmax=0.7,
            cbar_kws={"format": PercentFormatter(1.0)},
            ax=ax,
        )
        ax.axvline(WINDOW + 0.5, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{event} - {target}")
        ax.set_xlabel("Days from event")
        ax.set_ylabel("Year")
        ticks = np.arange(0, 2 * WINDOW + 1, 10)
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels([str(t - WINDOW) for t in ticks], rotation=0)

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary_bars(summary: pd.DataFrame, events: list[str], filename: str) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=summary[summary["event"].isin(events)],
        x="year",
        y="event_7d_pct_vs_local_baseline",
        hue="target",
        col="event",
        kind="bar",
        height=5,
        aspect=1.55,
        palette={"Revenue": "#1f77b4", "COGS": "#d62728"},
        sharey=True,
    )
    g.set_axis_labels("Year", "Mean % vs baseline, event window [-3,+3]")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.axhline(0, color="#333333", linewidth=1)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
        ax.tick_params(axis="x", rotation=45)
    out = OUT_DIR / filename
    g.figure.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(g.figure)
    return out


def plot_impact_ranking(aggregate: pd.DataFrame) -> Path:
    data = aggregate.sort_values("rank_score", ascending=True)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(
        data=data,
        y="event",
        x="event7_mean",
        hue="target",
        palette={"Revenue": "#1f77b4", "COGS": "#d62728"},
        ax=ax,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
    ax.set_xlabel("Mean % vs local baseline, event window [-3,+3]")
    ax.set_ylabel("")
    ax.set_title("Holiday/event impact ranking")
    fig.tight_layout()
    out = OUT_DIR / "holiday_all_events_impact_ranking.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_event_target_heatmap(aggregate: pd.DataFrame) -> Path:
    pivot = aggregate.pivot(index="event", columns="target", values="event7_mean")
    order = aggregate[["event", "rank_score"]].drop_duplicates().sort_values("rank_score", ascending=False)["event"]
    pivot = pivot.loc[order, TARGETS]
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        vmin=-0.55,
        vmax=0.55,
        annot=True,
        fmt=".0%",
        cbar_kws={"format": PercentFormatter(1.0)},
        ax=ax,
    )
    ax.set_title("Holiday/event mean impact by target")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    out = OUT_DIR / "holiday_all_events_target_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(summary: pd.DataFrame, paths: list[Path]) -> Path:
    aggregate = aggregate_summary(summary)
    top_events = aggregate[["event", "rank_score"]].drop_duplicates().head(8)["event"].tolist()

    report = OUT_DIR / "holiday_impact_summary.md"
    lines = [
        "# Holiday Impact Visualization Summary",
        "",
        "This analysis uses only `data/sales.csv` and `docs/vietnam_holiday_calendar_2012_2024.csv`.",
        "It does not use target values from `sample_submission.csv`.",
        "",
        "Impact is measured as percentage deviation from a local baseline: median target value from days 31-60 before/after each event.",
        "",
        "## Top event signals",
        "",
        ", ".join(top_events),
        "",
        "## Aggregate impact, ranked by absolute event-window effect",
        "",
        aggregate[
            [
                "event",
                "target",
                "day0_mean",
                "pre7_mean",
                "event7_mean",
                "post7_mean",
                "event7_std",
                "n_years",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Output charts",
        "",
    ]
    lines.extend(f"- `{path.relative_to(ROOT)}`" for path in paths)
    lines.extend(
        [
            "",
            "## Output tables",
            "",
            "- `model_thang/artifacts/holiday_impact/holiday_event_panel.csv`",
            "- `model_thang/artifacts/holiday_impact/holiday_impact_by_year.csv`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sales, holidays = load_data()
    panel = build_event_panel(sales, holidays)
    summary = summarize(panel)
    aggregate = aggregate_summary(summary)
    top_events = aggregate[["event", "rank_score"]].drop_duplicates().head(6)["event"].tolist()

    panel.to_csv(OUT_DIR / "holiday_event_panel.csv", index=False)
    summary.to_csv(OUT_DIR / "holiday_impact_by_year.csv", index=False)
    aggregate.to_csv(OUT_DIR / "holiday_impact_aggregate_ranked.csv", index=False)

    paths = [
        plot_impact_ranking(aggregate),
        plot_event_target_heatmap(aggregate),
        plot_event_study(
            panel,
            DRILLDOWN_EVENTS,
            "holiday_event_study_tet_hungkings.png",
            "Holiday event study: Tet and Hung Kings, mean impact across years",
        ),
        plot_heatmaps(
            panel,
            DRILLDOWN_EVENTS,
            "holiday_yearly_heatmap_tet_hungkings.png",
            "Year-by-year holiday impact heatmap: Tet and Hung Kings",
        ),
        plot_summary_bars(summary, DRILLDOWN_EVENTS, "holiday_event_window_yearly_bars.png"),
        plot_event_study(
            panel,
            top_events,
            "holiday_top_events_event_study.png",
            "Holiday event study: top-ranked events, mean impact across years",
        ),
    ]
    report = write_report(summary, paths)

    print(f"Wrote {report}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
