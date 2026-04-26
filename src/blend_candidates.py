"""Blend two submission CSVs with a given weight.

Useful for combining sample_shape-based submission with custom-shape submission
to reduce variance if their errors are uncorrelated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def blend(csv_a: Path, csv_b: Path, weight_a: float, out: Path) -> Path:
    a = pd.read_csv(csv_a, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    b = pd.read_csv(csv_b, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    merged = a.merge(b, on="Date", suffixes=("_a", "_b"))
    merged["Revenue"] = weight_a * merged["Revenue_a"] + (1 - weight_a) * merged["Revenue_b"]
    merged["COGS"] = weight_a * merged["COGS_a"] + (1 - weight_a) * merged["COGS_b"]
    out_df = merged[["Date", "Revenue", "COGS"]].copy()
    out_df["Revenue"] = out_df["Revenue"].round(2)
    out_df["COGS"] = out_df["COGS"].round(2)
    out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--weight-a", type=float, default=0.5)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = blend(Path(a.a), Path(a.b), a.weight_a, Path(a.out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
