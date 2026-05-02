FROM python:3.13-slim

LABEL maintainer="All-Rounder Rookie <datathon2026>"
LABEL description="Datathon 2026 — Revenue/COGS forecasting pipeline (train + generate submission.csv)"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && curl -LsSf https://astral.sh/uv/0.11.3/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen 2>&1

COPY src/ src/
COPY data/sales.csv data/sample_submission.csv data/orders.csv data/order_items.csv \
     data/payments.csv data/shipments.csv data/returns.csv data/reviews.csv \
     data/customers.csv data/products.csv data/promotions.csv data/geography.csv \
     data/inventory.csv data/web_traffic.csv data/

RUN mkdir -p docs/sales_forecast_submission \
             data/processed/sales_forecast_submission/artifacts/final_candidates \
             data/processed/sales_forecast_submission/artifacts/inference \
             data/processed/sales_forecast_submission/artifacts/saved_models/direct_factory

COPY run_pipeline.sh ./
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
