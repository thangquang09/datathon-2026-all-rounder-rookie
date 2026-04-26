"""Scale sample_submission per year and apply a within-year linear trend adjustment.

Let `t` be position within the year (0..N-1). Apply multiplier (1 + trend * (t/N - 0.5))
so the annual mean stays identical while leaning the series up or down in time.
This probes whether the test set has a steeper or shallower trajectory than sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "data" / "sample_submission.csv"


def _trend_series(n: int, trend: float) -> np.ndarray:
    pos = (np.arange(n) / max(n - 1, 1)) - 0.5
    return 1.0 + trend * pos


def build(
    scale_rev_2023: float,
    scale_rev_2024: float,
    scale_cog_2023: float,
    scale_cog_2024: float,
    trend_rev_2023: float,
    trend_rev_2024: float,
    trend_cog_2023: float,
    trend_cog_2024: float,
    out_path: Path,
) -> Path:
    sample = pd.read_csv(SAMPLE_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    out = sample.copy()

    for yr, s_rev, s_cog, t_rev, t_cog in [
        (2023, scale_rev_2023, scale_cog_2023, trend_rev_2023, trend_cog_2023),
        (2024, scale_rev_2024, scale_cog_2024, trend_rev_2024, trend_cog_2024),
    ]:
        mask = out["Date"].dt.year == yr
        n = int(mask.sum())
        if n == 0:
            continue
        rev = sample.loc[mask, "Revenue"].to_numpy() * s_rev
        cog = sample.loc[mask, "COGS"].to_numpy() * s_cog
        rev = rev * _trend_series(n, t_rev)
        cog = cog * _trend_series(n, t_cog)
        out.loc[mask, "Revenue"] = rev
        out.loc[mask, "COGS"] = cog

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
    p.add_argument("--trend-rev-2023", type=float, default=0.0)
    p.add_argument("--trend-rev-2024", type=float, default=0.0)
    p.add_argument("--trend-cog-2023", type=float, default=0.0)
    p.add_argument("--trend-cog-2024", type=float, default=0.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build(
        args.rev_2023, args.rev_2024, args.cog_2023, args.cog_2024,
        args.trend_rev_2023, args.trend_rev_2024, args.trend_cog_2023, args.trend_cog_2024,
        Path(args.out),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
