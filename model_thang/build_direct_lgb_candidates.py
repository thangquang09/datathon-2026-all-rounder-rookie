from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_thang.forecast_pipeline import TARGETS, export_submission, load_sales, normalise_yearly, yearly_level_targets

ARTIFACTS = ROOT / "model_thang" / "artifacts"
ADV_OUT = ARTIFACTS / "advanced_experiments"


def main() -> None:
    ADV_OUT.mkdir(parents=True, exist_ok=True)
    debug = pd.read_csv(ARTIFACTS / "direct_factory_component_debug.csv", parse_dates=["Date"])
    direct_lgb = pd.DataFrame(
        {
            "Date": debug["Date"],
            "Revenue": debug["Revenue_lgb"],
            "COGS": debug["COGS_lgb"],
        }
    )
    sales = load_sales()
    levels = {target: yearly_level_targets(sales, target, "regime_recovery") for target in TARGETS}
    direct_lgb = normalise_yearly(direct_lgb, levels)
    direct_path = ADV_OUT / "submission_direct_lgb_regime.csv"
    export_submission(direct_lgb, direct_path)

    base = pd.read_csv(ARTIFACTS / "m5b50300515.csv", parse_dates=["Date"])
    for w in (0.05, 0.10, 0.15, 0.20, 0.21, 0.25, 0.30):
        out = base[["Date"]].copy()
        for target in TARGETS:
            out[target] = (1.0 - w) * base[target].to_numpy() + w * direct_lgb[target].to_numpy()
        path = ADV_OUT / f"submission_m5_lgb_direct_blend_{int((1-w)*100):02d}_{int(w*100):02d}.csv"
        export_submission(out, path)
        print(path)
    print(direct_path)


if __name__ == "__main__":
    main()
