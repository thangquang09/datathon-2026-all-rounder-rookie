"""Per-target blend weights and level perturbations."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_v2"

LB_LEVELS = {
    "Revenue": {2023: 4_045_000, 2024: 4_865_000},
    "COGS":    {2023: 3_745_000, 2024: 4_265_000},
}

LB_LEVELS_HI = {
    "Revenue": {2023: 4_200_000, 2024: 5_050_000},
    "COGS":    {2023: 3_900_000, 2024: 4_440_000},
}

LB_LEVELS_LO = {
    "Revenue": {2023: 3_900_000, 2024: 4_700_000},
    "COGS":    {2023: 3_600_000, 2024: 4_100_000},
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
    v1 = pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv")
    v2 = pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv")
    v3 = pd.read_csv(ROOT / "outputs/final_v3/model_v3_raw.csv")

    n1 = normalise(v1, LB_LEVELS)
    n2 = normalise(v2, LB_LEVELS)
    n3 = normalise(v3, LB_LEVELS)
    base = pd.DataFrame({"Date": n1["Date"]})

    # Per-target blends
    pt1 = base.copy()
    pt1["Revenue"] = 0.6 * n1["Revenue"] + 0.3 * n2["Revenue"] + 0.1 * n3["Revenue"]
    pt1["COGS"]    = 0.3 * n1["COGS"]    + 0.5 * n2["COGS"]    + 0.2 * n3["COGS"]
    export(normalise(pt1, LB_LEVELS), OUT / "pt_rev60_cog30.csv")

    pt2 = base.copy()
    pt2["Revenue"] = 0.5 * n1["Revenue"] + 0.5 * n2["Revenue"]
    pt2["COGS"]    = 0.4 * n1["COGS"]    + 0.4 * n2["COGS"] + 0.2 * n3["COGS"]
    export(normalise(pt2, LB_LEVELS), OUT / "pt_rev50_cog40.csv")

    # Level perturbations on best blend (v1=0.5,v2=0.3,v3=0.2)
    bb = base.copy()
    bb["Revenue"] = 0.5 * n1["Revenue"] + 0.3 * n2["Revenue"] + 0.2 * n3["Revenue"]
    bb["COGS"]    = 0.5 * n1["COGS"]    + 0.3 * n2["COGS"]    + 0.2 * n3["COGS"]

    export(normalise(bb, LB_LEVELS_HI), OUT / "bb_levels_hi.csv")
    export(normalise(bb, LB_LEVELS_LO), OUT / "bb_levels_lo.csv")

    # Print summary
    for f in ["pt_rev60_cog30.csv", "pt_rev50_cog40.csv", "bb_levels_hi.csv", "bb_levels_lo.csv"]:
        d = pd.read_csv(OUT / f)
        d["year"] = pd.to_datetime(d["Date"]).dt.year
        m = d.groupby("year")[["Revenue", "COGS"]].mean().round(0)
        print(f"\n{f}:\n{m}")


if __name__ == "__main__":
    main()
