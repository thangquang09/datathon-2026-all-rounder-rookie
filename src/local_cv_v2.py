"""Local CV v2: apply best scale to each candidate shape and compare.

Key insight: on LB, best "scale" is not about matching overall level but
it's a property of the organizer's test labels. Locally we emulate by
re-scaling each candidate shape to the actual holdout yearly mean and
measuring residual error. The shape with lowest residual error after
optimal scaling wins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "sales.csv"


def load_holdout(horizon_days: int = 548) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(train)
    fit = train.iloc[: n - horizon_days].reset_index(drop=True)
    holdout = train.iloc[n - horizon_days :].reset_index(drop=True)
    return fit, holdout


def shape_doy_avg(fit: pd.DataFrame, target_dates: pd.Series, target: str, years: tuple[int, ...]) -> np.ndarray:
    subset = fit[fit["Date"].dt.year.isin(years)]
    shape = subset.groupby(subset["Date"].dt.dayofyear)[target].mean()
    doys = pd.to_datetime(target_dates).dt.dayofyear
    return doys.map(shape).fillna(shape.mean()).to_numpy()


def shape_doy_weighted(fit: pd.DataFrame, target_dates: pd.Series, target: str, years: tuple[int, ...], half_life: float) -> np.ndarray:
    sub = fit[fit["Date"].dt.year.isin(years)].copy()
    ref = sub["Date"].dt.year.max()
    sub["age"] = ref - sub["Date"].dt.year
    sub["w"] = 0.5 ** (sub["age"] / max(half_life, 1e-6))
    sub["doy"] = sub["Date"].dt.dayofyear
    shape = sub.groupby("doy").apply(lambda g: np.average(g[target], weights=g["w"]))
    doys = pd.to_datetime(target_dates).dt.dayofyear
    return doys.map(shape).fillna(shape.mean()).to_numpy()


def shape_lag_recursive_smooth(fit: pd.DataFrame, target_dates: pd.Series, target: str, window: int = 3) -> np.ndarray:
    """For each target date t, use mean of t-365 +/- window days."""
    df = fit.set_index("Date")[target]
    out = []
    for d in pd.to_datetime(target_dates):
        anchor = d - pd.Timedelta(days=365)
        vals = []
        for off in range(-window, window + 1):
            key = anchor + pd.Timedelta(days=off)
            if key in df.index:
                vals.append(df.loc[key])
        if not vals:
            out.append(np.nan)
        else:
            out.append(np.mean(vals))
    return np.array(out, dtype=float)


def shape_sample(target_dates: pd.Series, target: str) -> np.ndarray:
    # For holdout (2021-07..2022-12), we synthesize a "sample-like" shape by
    # averaging same-doy from the same fit period years. We use years
    # 2019-2020 (most analogous years).
    sales = pd.read_csv(TRAIN_FILE, parse_dates=["Date"])
    # Use pre-holdout years only
    ends = pd.Timestamp("2021-07-03")
    sub = sales[sales["Date"] < ends]
    years = [2019, 2020]
    sub2 = sub[sub["Date"].dt.year.isin(years)]
    shape = sub2.groupby(sub2["Date"].dt.dayofyear)[target].mean()
    doys = pd.to_datetime(target_dates).dt.dayofyear
    return doys.map(shape).fillna(shape.mean()).to_numpy()


def best_year_scale(pred: np.ndarray, actual: np.ndarray, years: pd.Series) -> tuple[dict[int, float], np.ndarray]:
    """For each year in the holdout, choose s that minimizes MSE(actual - s*pred)."""
    out = pred.copy()
    scales = {}
    for y in sorted(years.unique()):
        mask = (years == y).to_numpy()
        p = pred[mask]
        a = actual[mask]
        # Minimize MSE: s* = sum(p*a) / sum(p^2)
        if np.sum(p * p) == 0:
            s = 1.0
        else:
            s = float(np.sum(p * a) / np.sum(p * p))
        scales[int(y)] = s
        out[mask] = p * s
    return scales, out


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def apply_dow_boost(
    pred: np.ndarray, dates: pd.Series, fit: pd.DataFrame, target: str, weight: float, years: tuple[int, ...]
) -> np.ndarray:
    sub = fit[fit["Date"].dt.year.isin(years)]
    dow_mean = sub.groupby(sub["Date"].dt.dayofweek)[target].mean()
    mult = dow_mean / dow_mean.mean()
    dow = pd.to_datetime(dates).dt.dayofweek
    factor = 1.0 + weight * (dow.map(mult) - 1.0)
    # Normalize per-year so level preserved
    yrs = pd.to_datetime(dates).dt.year
    out = pred.copy().astype(float)
    for y in yrs.unique():
        mask = (yrs == y).to_numpy()
        f = factor[mask].to_numpy()
        f = f / f.mean()
        out[mask] = pred[mask] * f
    return out


def run_experiments() -> pd.DataFrame:
    fit, holdout = load_holdout(548)
    dates = holdout["Date"]
    actual_rev = holdout["Revenue"].to_numpy()
    actual_cog = holdout["COGS"].to_numpy()
    years = pd.to_datetime(dates).dt.year

    experiments = {
        "sample_proxy": {
            "rev": shape_sample(dates, "Revenue"),
            "cog": shape_sample(dates, "COGS"),
        },
        "doy_avg_2y_1920": {
            "rev": shape_doy_avg(fit, dates, "Revenue", (2019, 2020)),
            "cog": shape_doy_avg(fit, dates, "COGS", (2019, 2020)),
        },
        "doy_avg_3y_182021": {
            "rev": shape_doy_avg(fit, dates, "Revenue", (2018, 2019, 2020, 2021)),
            "cog": shape_doy_avg(fit, dates, "COGS", (2018, 2019, 2020, 2021)),
        },
        "doy_wtd_hl1_all": {
            "rev": shape_doy_weighted(fit, dates, "Revenue", (2015, 2016, 2017, 2018, 2019, 2020, 2021), 1.0),
            "cog": shape_doy_weighted(fit, dates, "COGS", (2015, 2016, 2017, 2018, 2019, 2020, 2021), 1.0),
        },
        "doy_wtd_hl2_all": {
            "rev": shape_doy_weighted(fit, dates, "Revenue", (2015, 2016, 2017, 2018, 2019, 2020, 2021), 2.0),
            "cog": shape_doy_weighted(fit, dates, "COGS", (2015, 2016, 2017, 2018, 2019, 2020, 2021), 2.0),
        },
        "lag_smooth_w3": {
            "rev": shape_lag_recursive_smooth(fit, dates, "Revenue", 3),
            "cog": shape_lag_recursive_smooth(fit, dates, "COGS", 3),
        },
        "lag_smooth_w7": {
            "rev": shape_lag_recursive_smooth(fit, dates, "Revenue", 7),
            "cog": shape_lag_recursive_smooth(fit, dates, "COGS", 7),
        },
    }

    # Test DoW-injected variants of sample_proxy
    rev0 = experiments["sample_proxy"]["rev"]
    cog0 = experiments["sample_proxy"]["cog"]
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        experiments[f"sample_dow_w{w}"] = {
            "rev": apply_dow_boost(rev0, dates, fit, "Revenue", w, (2019, 2020, 2021)),
            "cog": apply_dow_boost(cog0, dates, fit, "COGS", w, (2019, 2020, 2021)),
        }

    rows = []
    for name, preds in experiments.items():
        rev_scales, rev_scaled = best_year_scale(preds["rev"], actual_rev, years)
        cog_scales, cog_scaled = best_year_scale(preds["cog"], actual_cog, years)
        rm = metrics(actual_rev, rev_scaled)
        cm = metrics(actual_cog, cog_scaled)
        rows.append({
            "name": name,
            "rev_mae": rm["mae"],
            "cog_mae": cm["mae"],
            "rev_rmse": rm["rmse"],
            "cog_rmse": cm["rmse"],
            "rev_r2": rm["r2"],
            "cog_r2": cm["r2"],
            "lb_proxy": rm["mae"] + cm["mae"],
            "rev_scales": rev_scales,
            "cog_scales": cog_scales,
        })
    return pd.DataFrame(rows).sort_values("lb_proxy").reset_index(drop=True)


if __name__ == "__main__":
    df = run_experiments()
    print(df[["name", "rev_mae", "cog_mae", "lb_proxy", "rev_r2", "cog_r2"]].to_string(index=False))
    print()
    print("Best config scales:")
    print(df.iloc[0][["name", "rev_scales", "cog_scales"]])
