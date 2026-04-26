"""Scale the sample_submission and optionally re-inject a DoW shape.

Historical 2019-2022 data has clear day-of-week pattern (Tue peak, Sat trough),
but sample_submission appears smoothed flat on DoW. We blend a fractional
DoW multiplier back in while preserving per-year level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "data" / "sample_submission.csv"
SALES_FILE = ROOT / "data" / "sales.csv"


def build_dow_mult(target: str, years: tuple[int, ...] = (2019, 2020, 2021, 2022)) -> pd.Series:
    sales = pd.read_csv(SALES_FILE, parse_dates=["Date"])
    recent = sales[sales["Date"].dt.year.isin(years)]
    dow_mean = recent.groupby(recent["Date"].dt.dayofweek)[target].mean()
    return dow_mean / dow_mean.mean()


def build(
    scale_rev_2023: float,
    scale_rev_2024: float,
    scale_cog_2023: float,
    scale_cog_2024: float,
    dow_weight_rev: float,
    dow_weight_cog: float,
    out_path: Path,
    dow_years: tuple[int, ...] = (2019, 2020, 2021, 2022),
) -> Path:
    sample = pd.read_csv(SAMPLE_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    mult_rev = build_dow_mult("Revenue", dow_years)
    mult_cog = build_dow_mult("COGS", dow_years)

    out = sample.copy()
    mask_2023 = out["Date"].dt.year == 2023
    mask_2024 = out["Date"].dt.year == 2024

    out.loc[mask_2023, "Revenue"] = sample.loc[mask_2023, "Revenue"] * scale_rev_2023
    out.loc[mask_2024, "Revenue"] = sample.loc[mask_2024, "Revenue"] * scale_rev_2024
    out.loc[mask_2023, "COGS"] = sample.loc[mask_2023, "COGS"] * scale_cog_2023
    out.loc[mask_2024, "COGS"] = sample.loc[mask_2024, "COGS"] * scale_cog_2024

    dow = out["Date"].dt.dayofweek
    rev_factor = 1.0 + dow_weight_rev * (dow.map(mult_rev) - 1.0)
    cog_factor = 1.0 + dow_weight_cog * (dow.map(mult_cog) - 1.0)

    for yr, mask in [(2023, mask_2023), (2024, mask_2024)]:
        if not mask.any():
            continue
        rv_sub = rev_factor[mask]
        cg_sub = cog_factor[mask]
        rv_sub = rv_sub / rv_sub.mean()
        cg_sub = cg_sub / cg_sub.mean()
        out.loc[mask, "Revenue"] = out.loc[mask, "Revenue"].to_numpy() * rv_sub.to_numpy()
        out.loc[mask, "COGS"] = out.loc[mask, "COGS"].to_numpy() * cg_sub.to_numpy()

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
    p.add_argument("--dow-weight-rev", type=float, default=0.0)
    p.add_argument("--dow-weight-cog", type=float, default=0.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build(
        args.rev_2023,
        args.rev_2024,
        args.cog_2023,
        args.cog_2024,
        args.dow_weight_rev,
        args.dow_weight_cog,
        Path(args.out),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
