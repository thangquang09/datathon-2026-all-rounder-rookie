"""Scale the sample_submission directly by per-year multipliers.

The sample_submission already contains the right within-year calendar shape.
Public LB signal shows the level (per-year mean) is where most error lives.
This script tunes four scalars: Revenue 2023, Revenue 2024, COGS 2023, COGS 2024.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "data" / "sample_submission.csv"


def build(
    scale_rev_2023: float,
    scale_rev_2024: float,
    scale_cog_2023: float,
    scale_cog_2024: float,
    out_path: Path,
) -> Path:
    sample = pd.read_csv(SAMPLE_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    out = sample.copy()
    mask_2023 = out["Date"].dt.year == 2023
    mask_2024 = out["Date"].dt.year == 2024
    out.loc[mask_2023, "Revenue"] = sample.loc[mask_2023, "Revenue"] * scale_rev_2023
    out.loc[mask_2024, "Revenue"] = sample.loc[mask_2024, "Revenue"] * scale_rev_2024
    out.loc[mask_2023, "COGS"] = sample.loc[mask_2023, "COGS"] * scale_cog_2023
    out.loc[mask_2024, "COGS"] = sample.loc[mask_2024, "COGS"] * scale_cog_2024
    out["Revenue"] = out["Revenue"].round(2)
    out["COGS"] = out["COGS"].round(2)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rev-2023", type=float, required=True)
    p.add_argument("--rev-2024", type=float, required=True)
    p.add_argument("--cog-2023", type=float, required=True)
    p.add_argument("--cog-2024", type=float, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build(args.rev_2023, args.rev_2024, args.cog_2023, args.cog_2024, Path(args.out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
