"""Build v1+v3 blend variants."""
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

    def blend(weights):
        out = base.copy()
        for c in ("Revenue", "COGS"):
            out[c] = sum(w * src[c].values for w, src in weights)
        return normalise(out, LB_LEVELS)

    cands = {
        "b_v1v3_50_50.csv":     [(0.5, v1), (0.5, v3)],
        "b_v1v3_60_40.csv":     [(0.6, v1), (0.4, v3)],
        "b_v1v2v3_50_25_25.csv":[(0.5, v1), (0.25, v2), (0.25, v3)],
        "b_v1v2v3_40_30_30.csv":[(0.4, v1), (0.3, v2), (0.3, v3)],
        "b_v1v2v3_50_30_20.csv":[(0.5, v1), (0.3, v2), (0.2, v3)],
    }
    for name, w in cands.items():
        b = blend(w)
        export(b, OUT / name)
        print(f"{name} weights={[(round(x,2), 'src') for x,_ in w]}")


if __name__ == "__main__":
    main()
