from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe seasonal submissions for public-LB calibration."
    )
    parser.add_argument("--train", default="data/sales.csv")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lag", type=int, default=365)
    parser.add_argument("--scale-2023-revenue", type=float, required=True)
    parser.add_argument("--scale-2024-revenue", type=float, required=True)
    parser.add_argument("--scale-2023-cogs", type=float, required=True)
    parser.add_argument("--scale-2024-cogs", type=float, required=True)
    return parser.parse_args()


def recursive_seasonal(
    train: pd.DataFrame,
    forecast_dates: pd.Series,
    target: str,
    lag: int,
    scale_2023: float,
    scale_2024: float,
) -> list[float]:
    history = train[["Date", target]].copy()
    preds: list[float] = []

    for forecast_date in pd.to_datetime(forecast_dates):
        base = history.loc[
            history["Date"].eq(forecast_date - pd.Timedelta(days=lag)), target
        ]
        if base.empty:
            base = history.loc[
                history["Date"].eq(forecast_date - pd.Timedelta(days=lag - 1)),
                target,
            ]

        pred = (
            float(base.iloc[0])
            if not base.empty
            else float(history[target].tail(365).mean())
        )
        pred *= scale_2023 if forecast_date.year == 2023 else scale_2024
        pred = max(0.0, pred)
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [forecast_date], target: [pred]})],
            ignore_index=True,
        )

    return preds


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train, parse_dates=["Date"]).sort_values("Date")
    sample = pd.read_csv(args.sample, parse_dates=["Date"]).sort_values("Date")

    submission = sample[["Date"]].copy()
    submission["Revenue"] = recursive_seasonal(
        train,
        sample["Date"],
        "Revenue",
        args.lag,
        args.scale_2023_revenue,
        args.scale_2024_revenue,
    )
    submission["COGS"] = recursive_seasonal(
        train,
        sample["Date"],
        "COGS",
        args.lag,
        args.scale_2023_cogs,
        args.scale_2024_cogs,
    )
    submission[["Revenue", "COGS"]] = submission[["Revenue", "COGS"]].round(2)
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)

    print(f"Saved {len(submission)} rows to {out}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
