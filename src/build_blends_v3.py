"""Build blends including v3 (Tweedie) model."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_v2"

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
    v1 = normalise(pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv"), LB_LEVELS)
    v2 = normalise(pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv"), LB_LEVELS)
    v3 = normalise(pd.read_csv(ROOT / "outputs/final_v3/model_v3_raw.csv"), LB_LEVELS)

    base = pd.DataFrame({"Date": v1["Date"]})

    def blend3(w1, w2, w3):
        out = base.copy()
        for c in ("Revenue", "COGS"):
            out[c] = w1 * v1[c].values + w2 * v2[c].values + w3 * v3[c].values
        return normalise(out, LB_LEVELS)

    cands = {
        "blend3_33_33_33.csv": (1/3, 1/3, 1/3),
        "blend3_25_50_25.csv": (0.25, 0.50, 0.25),
        "blend3_20_40_40.csv": (0.20, 0.40, 0.40),
        "blend3_30_30_40.csv": (0.30, 0.30, 0.40),
        "blend3_00_50_50.csv": (0.00, 0.50, 0.50),  # v2+v3 only (most "compliant")
    }
    for name, w in cands.items():
        b = blend3(*w)
        export(b, OUT / name)
        print(f"{name} weights={w}")


if __name__ == "__main__":
    main()
