# Time-Series Ensemble Research Notes

Scope: map contest-winning time-series ensemble tactics to the Datathon daily
Revenue/COGS task without using `sample_submission` target values.

## External Lessons

Sources reviewed:

- M5 1st place Kaggle writeup:
  <https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/yeonjun-in-stu-1st-place-solution>
- M5 competition results paper:
  <https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf>
- G-Research Crypto 2nd place Kaggle writeup:
  <https://www.kaggle.com/competitions/g-research-crypto-forecasting/writeups/nathaniel-maddux-2nd-place-solution>
- G-Research solution summary:
  <https://kaggle.curtischong.me/competitions/G-Research-Crypto-Forecasting>

Transferable tactics:

1. Use several time-based validation windows and track both mean and variance.
   M5 used multiple trailing windows; G-Research used walk-forward grouped CV
   with a temporal gap. For this task, one split is too fragile because the
   2019 structural break and 2022 recovery can dominate a single holdout.

2. Prefer model diversity over seed diversity alone. M5 averaged LightGBM
   models trained at different hierarchy scopes and with recursive vs direct
   forecasting. For this task, the closest analog is a blend of:
   - compact exogenous LGBM,
   - log-target LGBM,
   - Tweedie positive-demand LGBM,
   - broad feature-family LGBM,
   - shape baselines using seasonal/day-of-year structure.

3. Keep the blend simple first. Equal or low-dimensional weighted averages are
   robust when validation variance is high. A complex stack is risky unless we
   have out-of-fold predictions for every component on identical folds.

4. Feature engineering matters more than exotic model classes. Useful feature
   families for this dataset are calendar, Vietnamese commerce events, target
   lags/rollings, product mix, promo/refund leakage proxies, inventory pressure,
   customer lifecycle, web traffic, payment/logistics/CX, and geography.

5. Leakage hygiene is the main constraint. Here, `sales.csv` is generated almost
   exactly from `order_items + products`; same-day item value or payment value is
   therefore a target leak during training. Safe versions are lagged features,
   train-cutoff climatology, and recursive target features.

## Local Synthetic-Generator Clues

Observed locally from train CSVs:

- `Revenue` equals daily sum of `quantity * unit_price` from order items up to
  floating-point noise.
- `COGS` equals daily sum of `quantity * product.cogs` up to rounding noise.
- The 2019 break is large: average daily Revenue drops from about 5.07M
  pre-break to about 3.01M post-break.
- Post-break monthly shape is a scaled version of pre-break shape. Correlation
  with the pre-break monthly profile is about 0.93-0.97 for 2019-2022.
- Strong yearly recurrence exists: lag 365 correlation is about 0.79 for
  Revenue; lag 730 is about 0.72.
- Web traffic has signal but is not enough by itself: sessions correlation with
  Revenue is about 0.32.

Implication: the generator is likely built as a deterministic daily commerce
panel with stable seasonal shape, a structural level regime, and operational
tables generated around the same latent demand. The safest high-signal strategy
is to model demand shape plus train-only level scenarios, then ensemble diverse
models that learn the same shape differently.

## Current Mapping

Implemented features already cover the high-value buckets:

- Calendar and events: weekday, month, day-of-year, Fourier terms, Vietnamese
  shopping/holiday flags.
- Target memory: lags 7/14/28/56/91/182/364/365/371/548/728/730 and rolling
  statistics.
- Exogenous climatology: orders, item/product mix, payments, web, returns,
  reviews, shipments, inventory.
- Model diversity: v1/v2/v3/v4 legacy families plus standalone model, seasonal
  and day-of-year baselines.
- Level scenarios: recent mean, YoY continuation, log-linear post-2019, and
  train-only regime recovery.

## Next Experiments

Priority order:

1. Keep v1-v4 regime-recovery blend as the current strongest no-leak candidate.
2. Add identical-fold out-of-fold predictions for v1-v4 so blend weights are
   chosen by validation rather than only by prior leaderboard history.
3. Add a direct 548-horizon model family: train rows are historical
   date-to-date mappings where the label is `y[t + horizon]`, reducing recursive
   drift for long horizons.
4. Add category-seasonal indices explicitly: Streetwear/Casual May, GenZ June,
   Outdoor December, based only on train history.
5. Add a margin-ratio post-model: forecast Revenue shape and a train-only
   COGS/Revenue ratio shape, then derive COGS as a constrained companion target.

