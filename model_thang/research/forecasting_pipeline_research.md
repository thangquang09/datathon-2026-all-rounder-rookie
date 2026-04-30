# Forecasting Pipeline Research Mapping

This note maps the diagrammed pipeline to the Datathon daily Revenue/COGS
forecasting task.  The central rule remains unchanged: never use
`sample_submission.csv` target values.

## Diagram Interpretation

The picture is a hybrid forecast factory:

1. **Training set, 2012-2022**
   - Build a dense daily panel.
   - Run EDA, preprocessing, leakage checks, outlier/regime analysis.
   - Create lag, rolling, calendar, holiday, exogenous, and latent-shape
     features.

2. **Test set, 2023-2024**
   - Build the same feature schema for future dates.
   - Future unknown exogenous values must be blank, lagged from train history,
     or replaced by train-only climatology.

3. **Base models**
   - Ridge: linear trend/seasonality and stable extrapolation.
   - LightGBM: nonlinear feature interactions and lag/exogenous effects.
   - Prophet: decomposable trend, weekly/yearly seasonalities, holidays/events.
   - Specialist models: models trained for a specific horizon, quarter, regime,
     product/category profile, or target relationship.

4. **Ensemble**
   - Combine diverse forecasts, preferably with simple or low-dimensional
     weights unless identical-fold OOF predictions support a stacker.

5. **Calibration**
   - Apply train-only level/bias correction by year, quarter, month, weekday,
     or target-ratio constraints.
   - Calibration must be fitted on historical validation errors or historical
     level/regime assumptions only.

## Research Lessons

### Forecast Combination

Forecasting competitions repeatedly show that combinations beat a single
selected model more often than not.  M4 and M5 results support simple and hybrid
ensembles, including statistical + machine-learning mixtures.  The
"forecast combination puzzle" is relevant here: a simple average can be hard to
beat because it reduces model-selection variance.

Useful mapping:

- Keep the current v1/v2/v3/v4 equal-ish blend as a strong baseline.
- Add model families only if they add error diversity, not just another seed.
- Track component correlations; a model with slightly worse CV can still help if
  its errors are different.

### Reduction to Regression

Modern ML forecasting often reduces forecasting to supervised regression:
create target lags, rolling transformations, date features, and optionally
exogenous features, then fit a tabular model.  MLForecast and sktime both
formalize this pattern.  sktime also distinguishes recursive, direct, dirrec,
and multi-output multi-step strategies.

Useful mapping:

- Current pipeline is mostly recursive.
- Add direct horizon models for the 548-day horizon:
  `label = y[t + h]`, features are known at `t`.
- Add quarter/horizon specialists:
  - Q1: Tet/January-February low season,
  - Q2: April-June peak,
  - Q3: summer/transition,
  - Q4: Black Friday/11-12/low December.

### Prophet-Style Decomposition

Prophet is useful as a decomposable baseline: trend + weekly/yearly seasonality
+ holidays + regressors.  It is not usually the best final model for rich
commerce panels, but its trend/seasonality components can diversify a LightGBM
ensemble or provide residual targets.

Useful mapping:

- Build Prophet-like deterministic features already present:
  Fourier weekly/yearly terms, VN events, monthly/weekday effects.
- A true Prophet component may be useful if calibrated conservatively:
  train Prophet on post-2019 daily Revenue/COGS, add holidays, then normalize to
  train-only regime levels.
- Better than raw Prophet may be residual learning:
  Prophet/seasonal baseline forecast first, LightGBM predicts residual.

### Ridge / Linear Models

Ridge is valuable because it extrapolates smoothly and shrinks many correlated
Fourier/calendar/exogenous features.  It can be weaker alone but useful in a
blend when LightGBM overfits spikes.

Useful mapping:

- Fit Ridge/ElasticNet on log target with:
  Fourier terms, month/dow, regime flag, lag365/730 ratios, and train-only
  climatology.
- Use Ridge as a "smooth shape" component, not a primary model.

### Specialist Models

Specialists are localized models trained on a subset or for a specific
condition.  Research on localized ensembles and FFORMA-style meta-learning
supports using series/features/regime characteristics to choose or weight
models.

Useful mapping:

- Quarter specialists: separate models/calibrators for Q1/Q2/Q3/Q4.
- Regime specialists: pre-2019 shape model, post-2019 level model, 2022
  recovery model.
- Target specialists: Revenue direct, COGS direct, and margin-ratio model.
- Horizon specialists: 1-182 days vs 183-548 days because recursive drift grows
  over a long horizon.
- Event specialists: Tet, 4-6 peak season, 11/11, 12/12, Black Friday.

## Concrete Next Pipeline

The next high-value implementation should be:

1. **OOF store**
   - Generate identical-fold validation predictions for every component:
     v1, v2, v3, v4, standalone, shape, Ridge, Prophet-like, direct-horizon.

2. **Blend search**
   - Search simple non-negative weights per target using fold-mean MAE.
   - Penalize high fold standard deviation and high component correlation.

3. **Quarter specialists**
   - Fit fold-local correction factors by quarter/month/dow using train-only
     validation residuals.
   - Export candidates with conservative shrinkage toward 1.0.

4. **Direct horizon model**
   - Build rows `(anchor_date, forecast_date, horizon)` from train history.
   - Features: anchor target lags/rollings, forecast calendar/event features,
     forecast DoY/month climatology, regime gap features.
   - Train LightGBM/Ridge on historical pseudo-horizons such as 365-548 days.

5. **Margin-ratio calibrator**
   - Forecast Revenue and daily COGS/Revenue ratio separately.
   - Combine with clipping based on historical margin bands by month/regime.

## Sources

- M4 competition practitioner view:
  <https://www.sciencedirect.com/science/article/pii/S0169207019301189>
- M4 competition main results:
  <https://www.sciencedirect.com/science/article/pii/S0169207019301128>
- Forecast combinations review:
  <https://arxiv.org/abs/2205.04216>
- MLForecast feature-engineering workflow:
  <https://www.nixtla.io/blog/automated-time-series-feature-engineering-with-mlforecast>
- sktime forecasting/reduction documentation:
  <https://www.sktime.net/en/stable/api_reference/forecasting.html>
- Prophet seasonality, holidays, regressors:
  <https://facebook.github.io/prophet/docs/seasonality%2C_holiday_effects%2C_and_regressors.html>
- Prophet paper:
  <https://fossies.org/linux/prophet/docs/static/prophet_paper_preprint.pdf>
- Multi-step forecasting strategy review:
  <https://arxiv.org/abs/1108.3259>

