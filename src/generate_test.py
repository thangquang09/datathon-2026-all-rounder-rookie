"""Generate a test-set submission using a chosen shape + per-year scales.

Supports the same shape family as src.shape_v2, but applied to the test
dates (2023-01-01 -> 2024-07-01).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.shape_v2 import (
    load_sample,
    load_train,
    shape_doy_weighted,
    shape_doy_dow_blend,
    shape_lag_multi_year,
    shape_sample,
)

ROOT = Path(__file__).resolve().parents[1]


def build(
    shape_name: str,
    scale_rev_2023: float,
    scale_rev_2024: float,
    scale_cog_2023: float,
    scale_cog_2024: float,
    out_path: Path,
    **kwargs,
) -> Path:
    train = load_train()
    sample = load_sample()
    dates = sample["Date"]

    shape_funcs = {
        "sample": lambda t: shape_sample(dates, t),
        "doy_19_20": lambda t: shape_doy_weighted(train, dates, t, (2019, 2020), 0.0, 0),
        "doy_19_20_s7": lambda t: shape_doy_weighted(train, dates, t, (2019, 2020), 0.0, 7),
        "doy_19_22": lambda t: shape_doy_weighted(train, dates, t, (2019, 2022), 0.0, 0),
        "doy_19_22_s3": lambda t: shape_doy_weighted(train, dates, t, (2019, 2022), 0.0, 3),
        "doy_19_22_s7": lambda t: shape_doy_weighted(train, dates, t, (2019, 2022), 0.0, 7),
        "doy_19_22_hl2_s3": lambda t: shape_doy_weighted(train, dates, t, (2019, 2022), 2.0, 3),
        "doy_19_22_hl1_s3": lambda t: shape_doy_weighted(train, dates, t, (2019, 2022), 1.0, 3),
        "doy_dow_blend_19_22_hl2_dw04_s3": lambda t: shape_doy_dow_blend(train, dates, t, (2019, 2022), 2.0, 3, 0.4),
        "doy_dow_blend_18_22_hl2_dw04_s3": lambda t: shape_doy_dow_blend(train, dates, t, (2018, 2022), 2.0, 3, 0.4),
        "doy_dow_blend_19_22_hl1_dw04_s3": lambda t: shape_doy_dow_blend(train, dates, t, (2019, 2022), 1.0, 3, 0.4),
        "doy_dow_blend_19_22_hl2_dw02_s5": lambda t: shape_doy_dow_blend(train, dates, t, (2019, 2022), 2.0, 5, 0.2),
        "doy_dow_blend_19_22_hl2_dw03_s3": lambda t: shape_doy_dow_blend(train, dates, t, (2019, 2022), 2.0, 3, 0.3),
        "lag_multi_year_hl2": lambda t: shape_lag_multi_year(train, dates, t, [1, 2, 3], 2.0),
        "sample_plus_doy_dow_blend": lambda t: (
            0.5 * shape_sample(dates, t)
            + 0.5 * shape_doy_dow_blend(train, dates, t, (2019, 2022), 2.0, 3, 0.4)
        ),
        "sample_plus_doy_19_22_s3": lambda t: (
            0.6 * shape_sample(dates, t)
            + 0.4 * shape_doy_weighted(train, dates, t, (2019, 2022), 0.0, 3)
        ),
    }

    if shape_name not in shape_funcs:
        raise ValueError(f"Unknown shape {shape_name}. Options: {sorted(shape_funcs)}")

    rev_pred = shape_funcs[shape_name]("Revenue")
    cog_pred = shape_funcs[shape_name]("COGS")

    out = sample[["Date"]].copy()
    years = out["Date"].dt.year
    rev_scaled = rev_pred.copy()
    cog_scaled = cog_pred.copy()
    mask_23 = (years == 2023).to_numpy()
    mask_24 = (years == 2024).to_numpy()
    rev_scaled[mask_23] = rev_pred[mask_23] * scale_rev_2023
    rev_scaled[mask_24] = rev_pred[mask_24] * scale_rev_2024
    cog_scaled[mask_23] = cog_pred[mask_23] * scale_cog_2023
    cog_scaled[mask_24] = cog_pred[mask_24] * scale_cog_2024

    out["Revenue"] = np.round(rev_scaled, 2)
    out["COGS"] = np.round(cog_scaled, 2)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True)
    p.add_argument("--rev-2023", type=float, required=True)
    p.add_argument("--rev-2024", type=float, required=True)
    p.add_argument("--cog-2023", type=float, required=True)
    p.add_argument("--cog-2024", type=float, required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build(
        args.shape,
        args.rev_2023,
        args.rev_2024,
        args.cog_2023,
        args.cog_2024,
        Path(args.out),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
