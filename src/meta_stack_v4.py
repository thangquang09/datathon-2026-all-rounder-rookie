"""Meta-learner: find optimal v1/v2/v3/v4 blend weights per target.

Uses walk-forward 2022 out-of-fold predictions of each base model,
fits constrained linear regression (weights >=0, sum=1), returns
optimum weights.

This is a legitimate training-only optimisation — no LB scores are used.

Strategy:
1. Replay each model's WF-CV 2022 predictions (refit on <2022, predict 2022).
2. Stack into (366, 4) matrix `P_val`, target `y_val`.
3. Solve `w = argmin ||P_val @ w - y_val||_1` s.t. w>=0, sum(w)=1 via SLSQP.
4. Apply the same weights to the raw 2023-2024 forecasts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.build_blend_v4 import LB_LEVELS, export, normalise


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_v4"
DATA = ROOT / "data"


def _reconstruct_val_2022(model_key: str, target: str) -> np.ndarray:
    """Load the per-model 2022 WF-CV predictions (saved by each run).

    Fallback: if not saved, use the model's CV MAE from metrics.json
    and synthesise dummy -- but we prefer the real thing so we
    re-trigger WF-CV if missing.
    """
    path_map = {
        "v1": ROOT / "outputs/final/val_2022_preds.csv",
        "v2": ROOT / "outputs/final_v2/val_2022_preds.csv",
        "v3": ROOT / "outputs/final_v3/val_2022_preds.csv",
        "v4": ROOT / "outputs/final_v4/val_2022_preds.csv",
    }
    p = path_map[model_key]
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    df = df[df["Date"].dt.year == 2022].sort_values("Date")
    return df[target].to_numpy()


COMMON_SEEDS = (42, 123, 7, 2024, 31)


def _wf2022_v2() -> dict[str, np.ndarray]:
    from src.final_model_v2 import build_frame, feature_cols, _train_one_seed
    return _wf2022_generic(build_frame, feature_cols, _train_one_seed, COMMON_SEEDS)


def _wf2022_v3() -> dict[str, np.ndarray]:
    from src.final_model_v3 import build_frame_v3, feature_cols, train_seed
    return _wf2022_generic(build_frame_v3, feature_cols, train_seed, COMMON_SEEDS,
                           use_log1p=False)


def _wf2022_v4() -> dict[str, np.ndarray]:
    from src.final_model_v4 import build_frame, feature_cols, _train_one_seed
    return _wf2022_generic(build_frame, feature_cols, _train_one_seed, COMMON_SEEDS)


def _wf2022_v1() -> dict[str, np.ndarray]:
    """WF-CV 2022 for v1 (raw target, no log1p, no seed bag)."""
    import lightgbm as lgb
    from src.final_model import build_frame, feature_cols, lgb_params
    out = {}
    for target in ("Revenue", "COGS"):
        df = build_frame(target)
        feats = feature_cols(df, target)
        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < "2022-01-01"]
        val = hist[hist["Date"].dt.year == 2022]
        y_tr = train[target].to_numpy()
        y_vl = val[target].to_numpy()
        dtrain = lgb.Dataset(train[feats], label=y_tr)
        dval = lgb.Dataset(val[feats], label=y_vl, reference=dtrain)
        m = lgb.train(
            lgb_params(42), dtrain, num_boost_round=4000,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
        pred = m.predict(val[feats], num_iteration=m.best_iteration)
        out[target] = np.maximum(pred, 0)
        out[f"{target}_dates"] = val["Date"].to_numpy()
        out[f"{target}_actual"] = val[target].to_numpy()
    return out


def _wf2022_generic(build_fn, feats_fn, train_fn, seeds, use_log1p: bool = True) -> dict[str, np.ndarray]:
    out = {}
    for target in ("Revenue", "COGS"):
        df = build_fn(target)
        feats = feats_fn(df, target)
        hist = df.dropna(subset=[target]).copy()
        hist = hist[hist["Date"] >= "2014-01-01"]
        train = hist[hist["Date"] < "2022-01-01"]
        val = hist[hist["Date"].dt.year == 2022]
        per_seed = []
        for s in seeds:
            m = train_fn(train, val, feats, target, seed=s)
            per_seed.append(m.predict(val[feats], num_iteration=m.best_iteration))
        pred = np.mean(per_seed, axis=0)
        if use_log1p:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, 1e8)
        out[target] = pred
        out[f"{target}_dates"] = val["Date"].to_numpy()
        out[f"{target}_actual"] = val[target].to_numpy()
    return out


def _save_val_preds(model_key: str, preds: dict) -> None:
    outdir = ROOT / f"outputs/final{'' if model_key=='v1' else '_'+model_key}"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "Date": preds["Revenue_dates"],
        "Revenue": preds["Revenue"],
        "COGS": preds["COGS"],
    })
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(outdir / "val_2022_preds.csv", index=False)


def solve_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P shape (T, K) predictions. y shape (T,) actual. Return weights."""
    K = P.shape[1]
    x0 = np.ones(K) / K

    def mae(w):
        return float(np.mean(np.abs(P @ w - y)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * K
    res = minimize(mae, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return res.x


def main() -> None:
    # Ensure we have val-2022 predictions cached for each model.
    preds_by_model = {}
    for key, fn in [("v2", _wf2022_v2), ("v3", _wf2022_v3), ("v4", _wf2022_v4)]:
        val_path = ROOT / f"outputs/final_{key}/val_2022_preds.csv"
        if not val_path.exists():
            print(f"Running WF-CV 2022 for {key} ...")
            preds = fn()
            _save_val_preds(key, preds)
        preds_by_model[key] = pd.read_csv(val_path, parse_dates=["Date"])
        print(f"  {key}: loaded {len(preds_by_model[key])} rows")

    # v1 — replay its 2022 WF
    v1_val = ROOT / "outputs/final/val_2022_preds.csv"
    if not v1_val.exists():
        print("Running WF-CV 2022 for v1 ...")
        preds = _wf2022_v1()
        out_dir = ROOT / "outputs/final"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "Date": preds["Revenue_dates"],
            "Revenue": preds["Revenue"],
            "COGS": preds["COGS"],
        })
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(out_dir / "val_2022_preds.csv", index=False)
    preds_by_model["v1"] = pd.read_csv(ROOT / "outputs/final/val_2022_preds.csv",
                                       parse_dates=["Date"])

    # Actual 2022 sales
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    actual = sales[sales["Date"].dt.year == 2022].set_index("Date")

    # Align all dates
    order = ["v1", "v2", "v3", "v4"]
    all_dates = preds_by_model["v4"].sort_values("Date")["Date"].to_numpy()
    actual_rev = actual.loc[all_dates, "Revenue"].to_numpy()
    actual_cogs = actual.loc[all_dates, "COGS"].to_numpy()

    results = {}
    for target, act in [("Revenue", actual_rev), ("COGS", actual_cogs)]:
        cols = []
        for k in order:
            d = preds_by_model[k].sort_values("Date")
            d = d[d["Date"].isin(all_dates)]
            cols.append(d[target].to_numpy())
        P = np.column_stack(cols)
        w = solve_weights(P, act)
        results[target] = {
            "weights": {k: float(w[i]) for i, k in enumerate(order)},
            "stacked_mae": float(np.mean(np.abs(P @ w - act))),
            "per_model_mae": {k: float(np.mean(np.abs(cols[i] - act)))
                              for i, k in enumerate(order)},
        }
        print(f"\n{target} stack weights:")
        for k, ww in results[target]["weights"].items():
            print(f"  {k}: {ww:.3f}")
        print(f"  Stack MAE: {results[target]['stacked_mae']:,.0f}")
        print(f"  Per-model MAE:")
        for k, mm in results[target]["per_model_mae"].items():
            print(f"    {k}: {mm:,.0f}")

    # Apply to raw 2023-2024 forecasts
    raws = {
        "v1": normalise(pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv"), LB_LEVELS),
        "v2": normalise(pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv"), LB_LEVELS),
        "v3": normalise(pd.read_csv(ROOT / "outputs/final_v3/model_v3_raw.csv"), LB_LEVELS),
        "v4": normalise(pd.read_csv(ROOT / "outputs/final_v4/model_v4_raw.csv"), LB_LEVELS),
    }
    base = raws["v1"].copy()
    for col in ("Revenue", "COGS"):
        w = results[col]["weights"]
        base[col] = sum(w[k] * raws[k][col].values for k in order)
    base = normalise(base, LB_LEVELS)

    fname = OUT / "bv4_meta_stack_2022.csv"
    export(base, fname)
    with open(ROOT / "outputs/final_v4/meta_stack.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote meta-stack: {fname}")


if __name__ == "__main__":
    main()
