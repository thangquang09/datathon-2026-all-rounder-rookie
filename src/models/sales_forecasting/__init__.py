"""Final sales forecasting pipeline package.

Code lives under `src/`; generated artifacts and the final submission live in
`data/processed/sales_forecast_submission/`; documentation lives in
`docs/sales_forecast_submission/`.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "processed" / "sales_forecast_submission" / "artifacts"
SUBMISSION_PATH = DATA_DIR / "processed" / "sales_forecast_submission" / "submission.csv"
DOCS_DIR = REPO_ROOT / "docs" / "sales_forecast_submission"

