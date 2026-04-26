"""Build v4 submission candidates.

Produces:
- v4 alone (LB-calibrated)
- v1+v2+v3+v4 blends with grid of weights around the current best
- v4 + margin-ratio hedge for COGS (COGS = Revenue * predicted_ratio
  renormalised to LB level)

All outputs are renormalised to `LB_LEVELS` for yearly mean consistency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_v4"
OUT.mkdir(parents=True, exist_ok=True)

LB_LEVELS = {
    "Revenue": {2023: 4_045_000.0, 2024: 4_865_000.0},
    "COGS":    {2023: 3_745_000.0, 2024: 4_265_000.0},
}


def normalise(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    yrs = out["Date"].dt.year
    for col, lv in levels.items():
        for y, want in lv.items():
            m = yrs == y
            have = out.loc[m, col].mean()
            if have > 0:
                out.loc[m, col] = out.loc[m, col] * (want / have)
    return out


def export(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out["Revenue"] = out["Revenue"].round(2)
    out["COGS"] = out["COGS"].round(2)
    assert len(out) == 548
    assert (out[["Revenue", "COGS"]] > 0).all().all()
    out.to_csv(path, index=False)
    print(f"  wrote {path.name}")


def main() -> None:
    v1 = normalise(pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv"), LB_LEVELS)
    v2 = normalise(pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv"), LB_LEVELS)
    v3 = normalise(pd.read_csv(ROOT / "outputs/final_v3/model_v3_raw.csv"), LB_LEVELS)
    v4 = normalise(pd.read_csv(ROOT / "outputs/final_v4/model_v4_raw.csv"), LB_LEVELS)

    for c in ("Revenue", "COGS"):
        for name, d in [("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4)]:
            if d[c].values.shape != v4[c].values.shape:
                raise AssertionError(f"shape mismatch {name} {c}")

    candidates = {
        "v4_only.csv":                           {"v1": 0.00, "v2": 0.00, "v3": 0.00, "v4": 1.00},
        "bv4_v1v2v3v4_40_25_15_20.csv":          {"v1": 0.40, "v2": 0.25, "v3": 0.15, "v4": 0.20},
        "bv4_v1v2v3v4_45_25_15_15.csv":          {"v1": 0.45, "v2": 0.25, "v3": 0.15, "v4": 0.15},
        "bv4_v1v2v3v4_35_25_15_25.csv":          {"v1": 0.35, "v2": 0.25, "v3": 0.15, "v4": 0.25},
        "bv4_v1v2v3v4_45_20_10_25.csv":          {"v1": 0.45, "v2": 0.20, "v3": 0.10, "v4": 0.25},
        "bv4_v1v4_60_40.csv":                    {"v1": 0.60, "v2": 0.00, "v3": 0.00, "v4": 0.40},
        "bv4_v1v4_50_50.csv":                    {"v1": 0.50, "v2": 0.00, "v3": 0.00, "v4": 0.50},
        "bv4_v2v4_50_50.csv":                    {"v1": 0.00, "v2": 0.50, "v3": 0.00, "v4": 0.50},
        "bv4_v1v2v3v4_50_25_15_10.csv":          {"v1": 0.50, "v2": 0.25, "v3": 0.15, "v4": 0.10},
        "bv4_ref_v1v2v3_50_30_20.csv":           {"v1": 0.50, "v2": 0.30, "v3": 0.20, "v4": 0.00},  # sanity reference
    }

    for fname, w in candidates.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights must sum to 1 for {fname}"
        base = v1.copy()
        for col in ("Revenue", "COGS"):
            base[col] = (
                w["v1"] * v1[col].values
                + w["v2"] * v2[col].values
                + w["v3"] * v3[col].values
                + w["v4"] * v4[col].values
            )
        base = normalise(base, LB_LEVELS)
        export(base, OUT / fname)

    # Margin-ratio hedge: use the best blend's Revenue but derive COGS
    # from the margin-ratio model.
    try:
        mr = pd.read_csv(ROOT / "outputs/final_v4/margin_ratio_raw.csv", parse_dates=["Date"])
        best_blend = pd.read_csv(OUT / "bv4_v1v2v3v4_45_25_15_15.csv", parse_dates=["Date"])
        hedged = best_blend.copy()
        hedged["COGS"] = hedged["Revenue"] * mr["margin_ratio"].values
        hedged = normalise(hedged, LB_LEVELS)
        export(hedged, OUT / "bv4_margin_hedge_cogs.csv")
    except FileNotFoundError:
        print("  skipped margin hedge (file missing)")

    print(f"\nTotal candidates: {len(list(OUT.glob('*.csv')))}")


if __name__ == "__main__":
    main()
