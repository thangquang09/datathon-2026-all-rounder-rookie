from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_thang.explainable_forecast_factory import deterministic_calendar  # noqa: E402
from model_thang.forecast_pipeline import (  # noqa: E402
    FORECAST_END,
    FORECAST_START,
    TARGETS,
    TRAIN_END,
    export_submission,
    load_sales,
    normalise_yearly,
    yearly_level_targets,
)


OUT = ROOT / "model_thang" / "artifacts" / "deep_sequence"
BASE_ARTIFACTS = ROOT / "model_thang" / "artifacts"

PAST = 730
HORIZON = 548
STRIDE_DAYS = 14
VAL_CUTOFF = pd.Timestamp("2020-12-31")
VAL_START = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-07-02")
SEED = 20260501


@dataclass
class Scalers:
    target_mean: np.ndarray
    target_std: np.ndarray
    cov_mean: np.ndarray
    cov_std: np.ndarray


@dataclass
class DatasetBundle:
    x_past: np.ndarray
    x_future: np.ndarray
    y: np.ndarray
    cutoffs: list[pd.Timestamp]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def prepare_sales_panel() -> pd.DataFrame:
    sales = load_sales().set_index("Date").sort_index()
    idx = pd.date_range(sales.index.min(), TRAIN_END, freq="D")
    sales = sales.reindex(idx)
    sales.index.name = "Date"
    if sales[list(TARGETS)].isna().any().any():
        sales[list(TARGETS)] = sales[list(TARGETS)].interpolate().ffill().bfill()
    return sales.reset_index()


def select_calendar_features(calendar: pd.DataFrame) -> list[str]:
    cols = []
    for col in calendar.columns:
        if not pd.api.types.is_numeric_dtype(calendar[col]):
            continue
        if col.startswith(("hol_", "vn_", "sin_", "cos_", "season_")):
            cols.append(col)
        elif col in {
            "forecast_year",
            "month",
            "day",
            "dow",
            "doy",
            "week",
            "quarter",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_payday_window",
            "is_midmonth_window",
        }:
            cols.append(col)
    return cols


def build_calendar_matrix(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    idx = pd.date_range(start, end, freq="D")
    cal = deterministic_calendar(idx)
    cols = select_calendar_features(cal)
    cal = cal[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cal, cols


def fit_scalers(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    train_end: pd.Timestamp,
) -> Scalers:
    hist = sales[sales["Date"] <= train_end].copy()
    y = np.log1p(hist[list(TARGETS)].to_numpy(dtype=np.float32))
    target_mean = y.mean(axis=0)
    target_std = y.std(axis=0)
    target_std[target_std < 1e-6] = 1.0

    cov = calendar.loc[:train_end].to_numpy(dtype=np.float32)
    cov_mean = np.nanmean(cov, axis=0)
    cov_std = np.nanstd(cov, axis=0)
    cov_std[cov_std < 1e-6] = 1.0
    return Scalers(target_mean, target_std, cov_mean, cov_std)


def normalize_targets(values: np.ndarray, scalers: Scalers) -> np.ndarray:
    return (np.log1p(np.clip(values, 0, None)) - scalers.target_mean) / scalers.target_std


def denormalize_targets(values: np.ndarray, scalers: Scalers) -> np.ndarray:
    return np.expm1(values * scalers.target_std + scalers.target_mean).clip(min=1.0)


def normalize_covariates(values: np.ndarray, scalers: Scalers) -> np.ndarray:
    return (values - scalers.cov_mean) / scalers.cov_std


def training_cutoffs(sales: pd.DataFrame, train_end: pd.Timestamp) -> list[pd.Timestamp]:
    min_date = sales["Date"].min() + pd.Timedelta(days=PAST - 1)
    max_cutoff = train_end - pd.Timedelta(days=HORIZON)
    return list(pd.date_range(min_date, max_cutoff, freq=f"{STRIDE_DAYS}D"))


def build_dataset(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    scalers: Scalers,
    cutoffs: list[pd.Timestamp],
    include_y: bool = True,
) -> DatasetBundle:
    y_series = sales.set_index("Date")[list(TARGETS)]
    x_past, x_future, y_out = [], [], []
    kept_cutoffs = []
    for cutoff in cutoffs:
        past_idx = pd.date_range(cutoff - pd.Timedelta(days=PAST - 1), cutoff, freq="D")
        future_idx = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        if past_idx.min() < y_series.index.min() or future_idx.max() > calendar.index.max():
            continue
        past = y_series.loc[past_idx].to_numpy(dtype=np.float32)
        if not np.isfinite(past).all():
            continue
        future_cov = calendar.loc[future_idx].to_numpy(dtype=np.float32)
        x_past.append(normalize_targets(past, scalers).T)
        x_future.append(normalize_covariates(future_cov, scalers))
        if include_y:
            if future_idx.max() > y_series.index.max():
                continue
            future_y = y_series.loc[future_idx].to_numpy(dtype=np.float32)
            y_out.append(normalize_targets(future_y, scalers))
        kept_cutoffs.append(cutoff)

    y_array = np.stack(y_out).astype(np.float32) if include_y else np.empty((len(x_past), HORIZON, 2), dtype=np.float32)
    return DatasetBundle(
        np.stack(x_past).astype(np.float32),
        np.stack(x_future).astype(np.float32),
        y_array,
        kept_cutoffs,
    )


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., -x.shape[-1] :]
        return x + out


class TCNDirect(nn.Module):
    def __init__(self, cov_dim: int, hidden: int = 48, dropout: float = 0.12) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(2, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden, 1, dropout),
            ResidualTCNBlock(hidden, 2, dropout),
            ResidualTCNBlock(hidden, 4, dropout),
            ResidualTCNBlock(hidden, 8, dropout),
        )
        self.future_proj = nn.Sequential(nn.Linear(cov_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(past)
        h = self.blocks(h)
        context = h[..., -1]
        future_h = self.future_proj(future)
        context = context[:, None, :].expand(-1, future.shape[1], -1)
        return self.head(torch.cat([context, future_h], dim=-1))


class MLPDirect(nn.Module):
    def __init__(self, cov_dim: int, hidden: int = 96, dropout: float = 0.20) -> None:
        super().__init__()
        self.past_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PAST * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_proj = nn.Sequential(nn.Linear(cov_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        context = self.past_encoder(past)
        future_h = self.future_proj(future)
        context = context[:, None, :].expand(-1, future.shape[1], -1)
        return self.head(torch.cat([context, future_h], dim=-1))


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    horizon_weight = torch.linspace(1.05, 0.95, pred.shape[1], device=pred.device)[None, :, None]
    return (nn.functional.smooth_l1_loss(pred, target, reduction="none") * horizon_weight).mean()


def train_model(
    model: nn.Module,
    data: DatasetBundle,
    device: torch.device,
    max_epochs: int = 120,
    patience: int = 15,
) -> tuple[nn.Module, dict[str, float]]:
    n = len(data.cutoffs)
    order = np.argsort(np.asarray(data.cutoffs, dtype="datetime64[ns]"))
    valid_n = max(8, int(n * 0.18))
    train_idx = order[:-valid_n]
    valid_idx = order[-valid_n:]

    train_ds = TensorDataset(
        torch.from_numpy(data.x_past[train_idx]),
        torch.from_numpy(data.x_future[train_idx]),
        torch.from_numpy(data.y[train_idx]),
    )
    valid_tensors = (
        torch.from_numpy(data.x_past[valid_idx]).to(device),
        torch.from_numpy(data.x_future[valid_idx]).to(device),
        torch.from_numpy(data.y[valid_idx]).to(device),
    )
    loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=12)

    best_state = None
    best_valid = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xp, xf, y in loader:
            xp, xf, y = xp.to(device), xf.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xp, xf), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_tensors[0], valid_tensors[1]), valid_tensors[2]).detach().cpu())
        scheduler.step(valid_loss)
        if valid_loss < best_valid - 1e-5:
            best_valid = valid_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_valid_loss": best_valid, "best_epoch": best_epoch, "n_examples": n}


def predict_bundle(model: nn.Module, data: DatasetBundle, scalers: Scalers, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(data.x_past).to(device),
            torch.from_numpy(data.x_future).to(device),
        ).detach().cpu().numpy()
    return denormalize_targets(pred[0], scalers)


def evaluate(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rows = {}
    for i, target in enumerate(TARGETS):
        err = actual[:, i] - pred[:, i]
        rows[f"{target}_mae"] = float(np.mean(np.abs(err)))
        rows[f"{target}_rmse"] = float(np.sqrt(np.mean(err**2)))
    rows["mean_mae"] = float(np.mean([rows["Revenue_mae"], rows["COGS_mae"]]))
    return rows


def run_training(train_end: pd.Timestamp, predict_cutoff: pd.Timestamp, predict_end: pd.Timestamp):
    sales = prepare_sales_panel()
    cal, cov_cols = build_calendar_matrix(sales["Date"].min(), FORECAST_END)
    scalers = fit_scalers(sales, cal, train_end)
    cutoffs = training_cutoffs(sales, train_end)
    train_data = build_dataset(sales, cal, scalers, cutoffs, include_y=True)
    pred_data = build_dataset(sales, cal, scalers, [predict_cutoff], include_y=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "tcn": TCNDirect(cov_dim=len(cov_cols)),
        "mlp": MLPDirect(cov_dim=len(cov_cols)),
    }
    trained = {}
    meta = {"device": str(device), "n_covariates": len(cov_cols), "covariates": cov_cols}
    for name, model in models.items():
        set_seed(SEED + (0 if name == "tcn" else 100))
        tic = time.time()
        print(f"training {name} with {len(train_data.cutoffs)} examples on {device}", flush=True)
        fitted, fit_meta = train_model(model, train_data, device)
        fit_meta["seconds"] = round(time.time() - tic, 2)
        print(f"finished {name}: {fit_meta}", flush=True)
        trained[name] = fitted
        meta[name] = fit_meta

    predictions = {name: predict_bundle(model, pred_data, scalers, device) for name, model in trained.items()}
    dates = pd.date_range(predict_cutoff + pd.Timedelta(days=1), predict_end, freq="D")
    if len(dates) != HORIZON:
        dates = pd.date_range(predict_cutoff + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    return sales, predictions, dates, meta, scalers, cal, pred_data, trained, device


def validation_actual(sales: pd.DataFrame) -> np.ndarray:
    return sales.set_index("Date").loc[pd.date_range(VAL_START, VAL_END, freq="D"), list(TARGETS)].to_numpy(dtype=float)


def tune_deep_weights(actual: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray]:
    best = (float("inf"), None, None)
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * preds["tcn"] + (1.0 - w) * preds["mlp"]
        score = evaluate(actual, blended)["mean_mae"]
        if score < best[0]:
            best = (score, {"tcn": float(w), "mlp": float(1.0 - w)}, blended)
    assert best[1] is not None and best[2] is not None
    return best[1], best[2]


def ablation_importance(
    model: nn.Module,
    pred_data: DatasetBundle,
    scalers: Scalers,
    actual: np.ndarray,
    cov_cols: list[str],
    device: torch.device,
) -> pd.DataFrame:
    base = predict_bundle(model, pred_data, scalers, device)
    base_mae = evaluate(actual, base)["mean_mae"]
    groups: dict[str, list[int]] = {
        "future_calendar": [
            i for i, c in enumerate(cov_cols)
            if not c.startswith(("hol_", "vn_", "season_")) and c not in {"forecast_year"}
        ],
        "future_holiday": [i for i, c in enumerate(cov_cols) if c.startswith(("hol_", "vn_", "season_"))],
        "future_vn_calendar": [i for i, c in enumerate(cov_cols) if c.startswith("vn_")],
        "future_regime_year": [i for i, c in enumerate(cov_cols) if c == "forecast_year"],
    }
    rows = []
    for group, idxs in groups.items():
        if not idxs:
            continue
        perturbed = DatasetBundle(pred_data.x_past.copy(), pred_data.x_future.copy(), pred_data.y, pred_data.cutoffs)
        perturbed.x_future[:, :, idxs] = 0.0
        pred = predict_bundle(model, perturbed, scalers, device)
        mae = evaluate(actual, pred)["mean_mae"]
        rows.append({"model": model.__class__.__name__, "group": group, "mean_mae": mae, "delta_mae": mae - base_mae})

    for channel, name in enumerate(["past_revenue", "past_cogs"]):
        perturbed = DatasetBundle(pred_data.x_past.copy(), pred_data.x_future.copy(), pred_data.y, pred_data.cutoffs)
        perturbed.x_past[:, channel, :] = 0.0
        pred = predict_bundle(model, perturbed, scalers, device)
        mae = evaluate(actual, pred)["mean_mae"]
        rows.append({"model": model.__class__.__name__, "group": name, "mean_mae": mae, "delta_mae": mae - base_mae})
    return pd.DataFrame(rows)


def plot_validation(dates: pd.DatetimeIndex, actual: np.ndarray, preds: dict[str, np.ndarray]) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for i, target in enumerate(TARGETS):
        axes[i].plot(dates, actual[:, i], color="#111111", label="actual", linewidth=1.3)
        for name, pred in preds.items():
            axes[i].plot(dates, pred[:, i], label=name, linewidth=1.0, alpha=0.85)
        axes[i].set_title(f"Deep direct validation forecast: {target}")
        axes[i].legend(loc="upper right")
    fig.tight_layout()
    path = OUT / "deep_validation_forecast.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(importance: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=importance, x="group", y="delta_mae", hue="model", ax=ax)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Deep model ablation importance on validation")
    ax.set_xlabel("")
    ax.set_ylabel("MAE increase when group is neutralized")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = OUT / "deep_ablation_importance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_submission(dates: pd.DatetimeIndex, pred: np.ndarray, path: Path, sales: pd.DataFrame) -> Path:
    df = pd.DataFrame({"Date": dates, "Revenue": pred[:, 0], "COGS": pred[:, 1]})
    levels = {target: yearly_level_targets(load_sales(), target, "regime_recovery") for target in TARGETS}
    df = normalise_yearly(df, levels)
    export_submission(df, path)
    return path


def write_report(metrics: pd.DataFrame, weights: dict[str, float], files: dict[str, str], meta: dict) -> Path:
    report = OUT / "deep_sequence_report.md"
    lines = [
        "# Deep Sequence Direct Model Report",
        "",
        "## Goal",
        "",
        "Test whether small direct deep-learning components add useful ensemble diversity without using future target leakage.",
        "",
        "## Architecture",
        "",
        "- `TCNDirect`: 1D dilated residual convolution encoder over the past 730 days, decoded with known future calendar/holiday covariates.",
        "- `MLPDirect`: N-BEATS-like direct MLP encoder over the past 730 days, decoded with known future calendar/holiday covariates.",
        "- Both models are multi-output and predict the full 548-day horizon for `Revenue` and `COGS` together.",
        "- No recursive test lag is used; no validation/test target appears in input features.",
        "",
        "## Validation",
        "",
        "- Cutoff: `2020-12-31`.",
        "- Validation horizon: `2021-01-01..2022-07-02`, 548 days.",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Deep ensemble weights chosen on validation:",
        "",
        pd.DataFrame([weights]).to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Training Metadata",
        "",
        "```json",
        json.dumps({k: v for k, v in meta.items() if k != "covariates"}, indent=2),
        "```",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{name}`: `{path}`" for name, path in files.items())
    lines.extend(
        [
            "",
            "## Kaggle Probe Status",
            "",
            "The first recommended probe is `submission_best_deep_blend_97_03.csv`.",
            "A previous upload attempt on 2026-04-30 returned `400 Bad Request` after upload, so the deep blend should not replace the current best until Kaggle returns a real score.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed()

    val_sales, val_preds, val_dates, val_meta, val_scalers, val_cal, val_pred_data, val_models, device = run_training(
        train_end=VAL_CUTOFF,
        predict_cutoff=VAL_CUTOFF,
        predict_end=VAL_END,
    )
    actual = validation_actual(val_sales)
    deep_weights, deep_val = tune_deep_weights(actual, val_preds)
    val_preds["deep_ensemble"] = deep_val

    metric_rows = []
    for name, pred in val_preds.items():
        row = {"model": name, **evaluate(actual, pred)}
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows).sort_values("mean_mae")
    metrics_df.to_csv(OUT / "deep_validation_metrics.csv", index=False)

    ablations = []
    cov_cols = val_meta["covariates"]
    for name, model in val_models.items():
        ab = ablation_importance(model, val_pred_data, val_scalers, actual, cov_cols, device)
        ab["model_name"] = name
        ablations.append(ab)
    ablation_df = pd.concat(ablations, ignore_index=True)
    ablation_df.to_csv(OUT / "deep_ablation_importance.csv", index=False)

    files = {
        "validation_metrics": str(OUT / "deep_validation_metrics.csv"),
        "ablation_importance": str(OUT / "deep_ablation_importance.csv"),
        "validation_plot": str(plot_validation(val_dates, actual, val_preds)),
        "ablation_plot": str(plot_ablation(ablation_df)),
    }

    final_sales, final_preds, final_dates, final_meta, *_ = run_training(
        train_end=TRAIN_END,
        predict_cutoff=TRAIN_END,
        predict_end=FORECAST_END,
    )
    deep_final = deep_weights["tcn"] * final_preds["tcn"] + deep_weights["mlp"] * final_preds["mlp"]
    files["direct_tcn"] = str(write_submission(final_dates, final_preds["tcn"], OUT / "submission_deep_tcn_regime.csv", final_sales))
    files["direct_mlp"] = str(write_submission(final_dates, final_preds["mlp"], OUT / "submission_deep_mlp_regime.csv", final_sales))
    files["direct_deep_ensemble"] = str(
        write_submission(final_dates, deep_final, OUT / "submission_deep_ensemble_regime.csv", final_sales)
    )

    base_path = BASE_ARTIFACTS / "advanced_experiments" / "submission_m5_lgb_direct_blend_80_20.csv"
    if base_path.exists():
        base = pd.read_csv(base_path, parse_dates=["Date"])
        deep_df = pd.read_csv(files["direct_deep_ensemble"], parse_dates=["Date"])
        for w in (0.03, 0.05, 0.08, 0.10):
            out = base[["Date"]].copy()
            for target in TARGETS:
                out[target] = (1 - w) * base[target].to_numpy(dtype=float) + w * deep_df[target].to_numpy(dtype=float)
            path = OUT / f"submission_best_deep_blend_{int((1-w)*100):02d}_{int(w*100):02d}.csv"
            export_submission(out, path)
            files[f"best_deep_{int((1-w)*100):02d}_{int(w*100):02d}"] = str(path)

    report = write_report(metrics_df, deep_weights, files, {**val_meta, "final": final_meta})
    files["report"] = str(report)
    (OUT / "deep_sequence_manifest.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
