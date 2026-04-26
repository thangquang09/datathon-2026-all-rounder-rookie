"""Stack the top-3 best LB submissions (all in the 739-740k band).

Rationale: each top candidate is already a well-tuned blend. Averaging
multiple near-optimal candidates reduces variance around the minimum
(a classic meta-ensemble trick). All candidates are pre-normalised to
LB_LEVELS so averaging preserves the yearly means.

Top-3 by public LB:
  b_v1v2v3_50_30_20.csv     739,471.96   (v1=0.50, v2=0.30, v3=0.20)
  best_pt_v1_tilt.csv       740,012.62   (per-target weights)
  best_blend_v1_45.csv      740,067.42   (v1=0.45, v2=0.30, v3=0.25)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDS = ROOT / "outputs" / "candidates_v2"
OUT = ROOT / "outputs" / "candidates_chronos"

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
    files = [
        "b_v1v2v3_50_30_20.csv",
        "best_pt_v1_tilt.csv",
        "best_blend_v1_45.csv",
    ]
    dfs = [pd.read_csv(CANDS / f) for f in files]

    base = pd.DataFrame({"Date": dfs[0]["Date"]})
    for c in ("Revenue", "COGS"):
        base[c] = sum(df[c].values for df in dfs) / len(dfs)

    b = normalise(base, LB_LEVELS)
    export(b, OUT / "stack_top3.csv")

    # Also build a weighted version favouring the #1
    wb = pd.DataFrame({"Date": dfs[0]["Date"]})
    for c in ("Revenue", "COGS"):
        wb[c] = 0.50 * dfs[0][c].values + 0.30 * dfs[1][c].values + 0.20 * dfs[2][c].values
    wb = normalise(wb, LB_LEVELS)
    export(wb, OUT / "stack_top3_w50_30_20.csv")

    # Pairwise correlation report
    for c in ("Revenue", "COGS"):
        r = pd.DataFrame({f: dfs[i][c] for i, f in enumerate(files)}).corr()
        print(f"=== {c} pairwise corr ===")
        print(r.round(4))
        print()


if __name__ == "__main__":
    main()
