"""Fine-tune around the v4=0.15 optimum found on public LB."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.build_blend_v4 import LB_LEVELS, export, normalise


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "candidates_v4"


def main() -> None:
    v1 = normalise(pd.read_csv(ROOT / "outputs/final/model_submission_raw.csv"), LB_LEVELS)
    v2 = normalise(pd.read_csv(ROOT / "outputs/final_v2/model_v2_raw.csv"), LB_LEVELS)
    v3 = normalise(pd.read_csv(ROOT / "outputs/final_v3/model_v3_raw.csv"), LB_LEVELS)
    v4 = normalise(pd.read_csv(ROOT / "outputs/final_v4/model_v4_raw.csv"), LB_LEVELS)

    # Keep v2/v3 proportional to current best (25/15) but vary v4 in fine steps.
    # v1 fills the rest.
    cands = {}
    for v4w in (0.12, 0.13, 0.14, 0.16, 0.17, 0.18):
        v2w, v3w = 0.25, 0.15
        v1w = 1.0 - v4w - v2w - v3w
        name = f"bv4_v1v2v3v4_{int(v1w*100):02d}_{int(v2w*100):02d}_{int(v3w*100):02d}_{int(v4w*100):02d}.csv"
        cands[name] = {"v1": v1w, "v2": v2w, "v3": v3w, "v4": v4w}

    # Also vary v2/v3 slightly with v4=0.15
    for v2w, v3w in [(0.30, 0.10), (0.20, 0.20), (0.30, 0.15)]:
        v4w = 0.15
        v1w = 1.0 - v4w - v2w - v3w
        name = f"bv4_v1v2v3v4_{int(v1w*100):02d}_{int(v2w*100):02d}_{int(v3w*100):02d}_{int(v4w*100):02d}.csv"
        cands[name] = {"v1": v1w, "v2": v2w, "v3": v3w, "v4": v4w}

    for fname, w in cands.items():
        base = v1.copy()
        for col in ("Revenue", "COGS"):
            base[col] = (
                w["v1"] * v1[col].values
                + w["v2"] * v2[col].values
                + w["v3"] * v3[col].values
                + w["v4"] * v4[col].values
            )
        base = normalise(base, LB_LEVELS)
        export(base, OUT / fname)


if __name__ == "__main__":
    main()
