#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo " Datathon 2026 — All-Rounder Rookie"
echo " Full Training + Submission Pipeline"
echo "============================================"

echo ""
START_TIME=$(date +%s)

echo "Running full train-save-infer-blend pipeline..."
uv run python -m src.models.sales_forecasting.train_save_infer_blend --skip-visuals 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo " DONE in ${ELAPSED}s!"
echo " submission.csv is ready at:"
echo " data/processed/sales_forecast_submission/submission.csv"
echo "============================================"
