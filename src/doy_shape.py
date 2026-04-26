"""Build per-year scaled submissions from day-of-year average shape.

Key idea: the sample_submission itself appears to be the DoY mean of
Revenue/COGS across 2019-2022. We reproduce that but with:

- Configurable reference years window.
- Day-of-week / day-of-year weight options.
- Matching calendar dow for the target date (shift by 1 day if needed).
- Per-year scale tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "sales.csv"
SAMPLE_FILE = ROOT / "data" / "sample_submission.csv"


def day_of_year_shape(train: pd.DataFrame, target: str, years_window: tuple[int, int]) -> pd.Series:
    """Mean per day-of-year across the given year window, returned indexed by DoY 1..366."""
    subset = train[(train["Date"].dt.year >= years_window[0]) & (train["Date"].dt.year <= years_window[1])].copy()
    subset["doy"] = subset["Date"].dt.dayofyear
    return subset.groupby("doy")[target].mean()


def weighted_day_of_year(train: pd.DataFrame, target: str, years_window: tuple[int, int], half_life_years: float) -> pd.Series:
    """Exponentially-weighted DoY mean: recent years get more weight."""
    subset = train[(train["Date"].dt.year >= years_window[0]) & (train["Date"].dt.year <= years_window[1])].copy()
    ref_year = years_window[1]
    subset["age"] = ref_year - subset["Date"].dt.year
    subset["weight"] = 0.5 ** (subset["age"] / max(half_life_years, 1e-6))
    subset["doy"] = subset["Date"].dt.dayofyear
    agg = subset.groupby("doy").apply(lambda g: np.average(g[target], weights=g["weight"]))
    return agg


def build_forecast_series(
    sample: pd.DataFrame,
    shape: pd.Series,
    target: str,
) -> pd.Series:
    """Map each forecast date to its doy and look up the shape."""
    doys = sample["Date"].dt.dayofyear
    return doys.map(shape).astype(float)


def apply_year_scale(series: pd.Series, years: pd.Series, scales: dict[int, float]) -> pd.Series:
    out = series.copy()
    for y, s in scales.items():
        mask = years == y
        out.loc[mask] = series.loc[mask] * s
    return out


def build_submission(
    years_window: tuple[int, int],
    half_life_years: float,
    scale_rev_2023: float,
    scale_rev_2024: float,
    scale_cog_2023: float,
    scale_cog_2024: float,
    out_path: Path,
) -> Path:
    train = pd.read_csv(TRAIN_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sample = pd.read_csv(SAMPLE_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    if half_life_years > 0:
        rev_shape = weighted_day_of_year(train, "Revenue", years_window, half_life_years)
        cog_shape = weighted_day_of_year(train, "COGS", years_window, half_life_years)
    else:
        rev_shape = day_of_year_shape(train, "Revenue", years_window)
        cog_shape = day_of_year_shape(train, "COGS", years_window)

    submission = sample[["Date"]].copy()
    submission["Revenue"] = build_forecast_series(sample, rev_shape, "Revenue")
    submission["COGS"] = build_forecast_series(sample, cog_shape, "COGS")

    years = submission["Date"].dt.year
    submission["Revenue"] = apply_year_scale(
        submission["Revenue"], years, {2023: scale_rev_2023, 2024: scale_rev_2024}
    )
    submission["COGS"] = apply_year_scale(
        submission["COGS"], years, {2023: scale_cog_2023, 2024: scale_cog_2024}
    )

    submission["Revenue"] = submission["Revenue"].round(2)
    submission["COGS"] = submission["COGS"].round(2)
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--years-from", type=int, default=2019)
    p.add_argument("--years-to", type=int, default=2022)
    p.add_argument("--half-life", type=float, default=0.0, help="0 = uniform weight; >0 enables exp decay")
    p.add_argument("--rev-2023", type=float, required=True)
    p.add_argument("--rev-2024", type=float, required=True)
    p.add_argument("--cog-2023", type=float, required=True)
    p.add_argument("--cog-2024", type=float, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build_submission(
        (args.years_from, args.years_to),
        args.half_life,
        args.rev_2023,
        args.rev_2024,
        args.cog_2023,
        args.cog_2024,
        Path(args.out),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
