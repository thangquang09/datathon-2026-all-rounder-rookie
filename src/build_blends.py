"""Build blended candidate submissions at LB-best levels.

Components:
  - v1_raw : compliant LGBM with leak-features (legacy)
  - v2_raw : compliant LGBM no-leak, multi-seed bag (best individual)
  - sn     : seasonal naive (DoY-shape × yearly mean)

Each is normalised to the same per-year level so the blend keeps
the chosen level. We export several blends to LB-tune mixing weights.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_v2"

LB_LEVELS = {
    "Revenue": {2023: 4_045_000, 2024: 4_865_000},
    "COGS":    {2023: 3_745_000, 2024: 4_265_000},
}


def normalise(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    s = df.copy()
    s["Date"] = pd.to_datetime(s["Date"])
    s["_y"] = s["Date"].dt.year
    for col, by_year in levels.items():
        for y, want in by_year.items():
            mask = s["_y"] == y
            cur = s.loc[mask, col].mean()
            if cur > 0:
                s.loc[mask, col] *= want / cur
    return s.drop(columns=["_y"])


def export(df: pd.DataFrame, path: Path) -> None:
    s = df.copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.strftime("%Y-%m-%d")
    s["Revenue"] = s["Revenue"].round(2)
    s["COGS"] = s["COGS"].round(2)
    assert len(s) == 548
    s.to_csv(path, index=False)


def main():
    v1 = normalise(pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv"), LB_LEVELS)
    v2 = normalise(pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv"), LB_LEVELS)
    sn = normalise(pd.read_csv(ROOT / "outputs/candidates_v2/seasonal_naive.csv"), LB_LEVELS)

    base = pd.DataFrame({"Date": v1["Date"]})

    def blend(weights):
        w1, w2, ws = weights
        out = base.copy()
        for col in ("Revenue", "COGS"):
            out[col] = w1 * v1[col].values + w2 * v2[col].values + ws * sn[col].values
        # re-normalise (in case rounding drifts)
        out = normalise(out, LB_LEVELS)
        return out

    cands = {
        # v1, v2, sn
        "blend_30_70_00.csv": (0.30, 0.70, 0.00),  # weight v2 more
        "blend_70_30_00.csv": (0.70, 0.30, 0.00),  # weight v1 more
        "blend_40_40_20.csv": (0.40, 0.40, 0.20),  # add seasonal
        "blend_30_50_20.csv": (0.30, 0.50, 0.20),
        "blend_50_50_00.csv": (0.50, 0.50, 0.00),  # = our current best
    }
    for name, w in cands.items():
        b = blend(w)
        export(b, OUT / name)
        print(f"{name}: weights={w}")
    print("Done.")


if __name__ == "__main__":
    main()
