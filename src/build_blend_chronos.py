"""Build blends that include Chronos-2 zero-shot forecast.

We mix Chronos-2 with the best compliant v1+v2+v3 LightGBM ensemble
to add foundation-model diversity. All candidates are renormalised to
the LB_LEVELS yearly means (same calibration used for previous 739k
submission).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_chronos"
OUT.mkdir(parents=True, exist_ok=True)

LB_LEVELS = {
    "Revenue": {2023: 4_045_000, 2024: 4_865_000},
    "COGS":    {2023: 3_745_000, 2024: 4_265_000},
}


def normalise(df, levels):
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


def export(df, path):
    s = df.copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.strftime("%Y-%m-%d")
    s["Revenue"] = s["Revenue"].round(2)
    s["COGS"] = s["COGS"].round(2)
    assert len(s) == 548
    s.to_csv(path, index=False)


def main():
    v1_raw = pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv")
    v2_raw = pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv")
    v3_raw = pd.read_csv(ROOT / "outputs/final_v3/model_v3_raw.csv")
    ch_raw = pd.read_csv(ROOT / "outputs/chronos/chronos2_raw.csv")

    v1 = normalise(v1_raw, LB_LEVELS)
    v2 = normalise(v2_raw, LB_LEVELS)
    v3 = normalise(v3_raw, LB_LEVELS)
    ch = normalise(ch_raw, LB_LEVELS)

    base = pd.DataFrame({"Date": v1["Date"]})

    def blend(weights):
        out = base.copy()
        for c in ("Revenue", "COGS"):
            vals = (
                weights["v1"] * v1[c].values
                + weights["v2"] * v2[c].values
                + weights["v3"] * v3[c].values
                + weights["ch"] * ch[c].values
            )
            out[c] = vals
        return normalise(out, LB_LEVELS)

    cands = {
        # Baseline reference: current best v1v2v3 (no chronos) => same as 739k submission
        "ref_v1v2v3_50_30_20.csv": {"v1": 0.50, "v2": 0.30, "v3": 0.20, "ch": 0.00},

        # Light chronos (safer, small diversity injection)
        "chr_15_v1v2v3_42_26_17.csv": {"v1": 0.425, "v2": 0.255, "v3": 0.17, "ch": 0.15},

        # Moderate chronos
        "chr_25_v1v2v3_37_22_15.csv": {"v1": 0.375, "v2": 0.225, "v3": 0.15, "ch": 0.25},

        # Heavy chronos
        "chr_40_v1v2v3_30_18_12.csv": {"v1": 0.30, "v2": 0.18, "v3": 0.12, "ch": 0.40},

        # Pure chronos (sanity)
        "chr_100.csv": {"v1": 0.0, "v2": 0.0, "v3": 0.0, "ch": 1.0},

        # Balanced 4-way
        "chr_4way_25.csv": {"v1": 0.25, "v2": 0.25, "v3": 0.25, "ch": 0.25},
    }

    for name, w in cands.items():
        b = blend(w)
        export(b, OUT / name)
        # Quick sanity check on yearly means
        bb = b.copy()
        bb["y"] = pd.to_datetime(bb["Date"]).dt.year
        print(name, "weights=", w)
        print("  ", bb.groupby("y")[["Revenue", "COGS"]].mean().round(0).to_dict())


if __name__ == "__main__":
    main()
