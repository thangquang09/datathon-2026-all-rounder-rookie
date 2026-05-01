from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
ARTIFACTS = ROOT / "artifacts"
SHAP_PATH = ARTIFACTS / "direct_factory_shap_importance.csv"
GAIN_PATH = ARTIFACTS / "direct_factory_feature_importance.csv"
OUT_DIR = ARTIFACTS / "figures"


def shorten_feature(name: str, max_len: int = 54) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def plot_top30_shap(shap: pd.DataFrame) -> Path:
    overall = (
        shap.groupby(["feature", "group"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .head(30)
        .sort_values("mean_abs_shap", ascending=True)
    )
    overall["feature_short"] = overall["feature"].map(shorten_feature)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.barplot(
        data=overall,
        y="feature_short",
        x="mean_abs_shap",
        hue="group",
        dodge=False,
        ax=ax,
    )
    ax.set_title("Top 30 features by SHAP importance, Revenue + COGS")
    ax.set_xlabel("Sum of mean absolute SHAP across targets")
    ax.set_ylabel("")
    ax.legend(title="Feature group", loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "top30_features_shap_overall.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_top30_by_target(shap: pd.DataFrame) -> Path:
    top = (
        shap.sort_values(["target", "mean_abs_shap"], ascending=[True, False])
        .groupby("target", group_keys=False)
        .head(30)
        .copy()
    )
    top["feature_short"] = top["feature"].map(shorten_feature)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(18, 13), sharex=False)
    for ax, target in zip(axes, ["Revenue", "COGS"]):
        g = top[top["target"].eq(target)].sort_values("mean_abs_shap", ascending=True)
        sns.barplot(
            data=g,
            y="feature_short",
            x="mean_abs_shap",
            hue="group",
            dodge=False,
            ax=ax,
        )
        ax.set_title(f"Top 30 SHAP features: {target}")
        ax.set_xlabel("Mean absolute SHAP")
        ax.set_ylabel("")
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Feature group", loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = OUT_DIR / "top30_features_shap_by_target.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_top30_gain(gain: pd.DataFrame) -> Path:
    overall = (
        gain.groupby(["feature", "group"], as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=False)
        .head(30)
        .sort_values("gain", ascending=True)
    )
    overall["feature_short"] = overall["feature"].map(shorten_feature)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.barplot(
        data=overall,
        y="feature_short",
        x="gain",
        hue="group",
        dodge=False,
        ax=ax,
    )
    ax.set_title("Top 30 features by LightGBM gain, Revenue + COGS")
    ax.set_xlabel("Sum of LightGBM gain across targets")
    ax.set_ylabel("")
    ax.legend(title="Feature group", loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "top30_features_gain_overall.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def export_shortlist(shap: pd.DataFrame, gain: pd.DataFrame) -> Path:
    shap_rank = (
        shap.groupby(["feature", "group"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    gain_rank = (
        gain.groupby("feature", as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=False)
    )
    out = shap_rank.merge(gain_rank, on="feature", how="left").head(30)
    out.insert(0, "rank", range(1, len(out) + 1))
    path = OUT_DIR / "top30_feature_shortlist.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shap = pd.read_csv(SHAP_PATH)
    gain = pd.read_csv(GAIN_PATH)

    paths = [
        plot_top30_shap(shap),
        plot_top30_by_target(shap),
        plot_top30_gain(gain),
        export_shortlist(shap, gain),
    ]
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
