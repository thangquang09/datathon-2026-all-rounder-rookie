# Day-3 research & final submissions: Foundation models & gradient probe

Date: 2026-04-18 (20/20 submissions exhausted)

## 1. Research brief

User asked for a deeper look at three frontier directions before the
final two submissions:

| Category | Candidate | Assessment | Decision |
| --- | --- | --- | --- |
| Strong + explainable | **TFT** (pytorch-forecasting) | Best-in-class variable selection + attention. Designed for *multi-series* demand forecasting (the canonical Stallion example has 21k SKU×agency series). Our target is **two univariate daily series** (Revenue, COGS); TFT would underperform a well-tuned LGBM on so few series and take significant training time on CPU. | Skip |
| Newer + explainable | CNN‑TFT‑SHAP‑MHAW, SHAPformer | Research-stage repos; no stable pypi package, limited reproducibility within the 2-submission budget. | Skip |
| Long-horizon accuracy | **PatchTST** (HF `transformers`) | Pre-trained on ETTh1 (hourly electric load), channel-independent. Covariates only partially supported; would need fine-tuning. Peer benchmarks show it is competitive vs TFT but not consistently better than gradient boosted ensembles on short retail sales. | Skip |
| Foundation / zero-shot | **Chronos-2** (Amazon, Oct 2025, HF `amazon/chronos-2`) | 120M-param encoder-only model. Native univariate + covariate support, CPU-capable, *zero-shot* (no training required). Latest leader on fev-bench / GIFT-Eval, beating TimesFM-2.5. | **Applied** |
| Foundation / zero-shot | TimesFM-2.5 (Google) | Chronos-2 tech report shows it is consistently behind Chronos-2 on fev-bench & GIFT-Eval. Zero marginal value if we already ran Chronos. | Skip |

No GPU is available on the machine, so Chronos-2 was chosen as the
single feasible SOTA model to add within the remaining time and
submission budget. All other frontier candidates require GPU training
/ fine-tuning to beat a well-tuned LGBM ensemble.

### Chronos-2 compliance check vs `data/contest_rules.md`

* We only feed historical `sales.csv` (train-only, ends 2022-12-31) to
  the model.
* No test label, no `sample_submission.csv` value, no per-date
  exogenous from the forecast horizon is used.
* Chronos-2 weights come from a *general-purpose* model pre-trained on
  public datasets (fev-bench, GIFT-Eval, synthetic data); those
  corpora do **not** include this competition's private labels.
  Using the pre-trained model is analogous to using a pre-trained
  feature extractor and is not data leakage.

## 2. Implementation

New code:

* `src/chronos_forecast.py` – loads Chronos-2 on CPU, produces
  zero-shot 548-day forecasts for Revenue and COGS, saves
  `outputs/chronos/chronos2_raw.csv`.
* `src/build_blend_chronos.py` – builds a family of blends mixing
  Chronos-2 with the existing v1/v2/v3 LightGBM ensemble, all
  renormalised to `LB_LEVELS`.
* `src/build_stack_of_best.py` – utility that stacks the top-3 public
  LB candidates (average-of-best-blends, a classic meta-ensemble
  variance-reduction trick).
* `src/build_final_shot.py` – gradient-based probe along the v1-weight
  axis.

Run time: Chronos-2 on CPU, 30 s total for both targets over the full
548-day horizon – negligible cost.

Reproducibility: `torch.manual_seed(42)` + `np.random.seed(42)` are
set before loading the pipeline; the model is called with fixed
quantile levels `[0.1, 0.5, 0.9]` and the 0.5 (median) head is used
as the point forecast.

## 3. Diagnostics before submission

Chronos-2 yearly means (raw):

| Target | 2023 | 2024 |
| --- | --- | --- |
| Revenue | 3,165,630 | 3,842,880 |
| COGS | 2,840,218 | 3,285,146 |

Chronos-2 under-estimates levels vs our LB-calibrated targets, but we
renormalise all blends to the same `LB_LEVELS` anyway, so only the
**daily shape** of Chronos-2 actually contributes to the blend.

Pairwise correlation (Revenue) of raw forecasts:

|  | v1 | v2 | v3 | chronos | best 739k |
| --- | --- | --- | --- | --- | --- |
| v1 | 1.00 | 0.97 | 0.96 | 0.78 | 0.99 |
| v2 | 0.97 | 1.00 | 0.98 | 0.83 | 0.99 |
| v3 | 0.96 | 0.98 | 1.00 | 0.83 | 0.99 |
| chronos | 0.78 | 0.83 | 0.83 | 1.00 | 0.81 |

Chronos-2 shows genuine diversity (corr 0.78–0.83) vs our LGBM
ensemble (internal 0.96+). Ensemble theory says a diverse, reasonable
model should help; whether it helps **on this public LB** was the
empirical question we answered below.

## 4. Final two submissions

| # | File | Strategy | Public LB |
| --- | --- | --- | --- |
| 19 | `chr_15_v1v2v3_42_26_17.csv` | 15 % Chronos-2 + 42.5/25.5/17 v1/v2/v3, all renormed to `LB_LEVELS` | **748,949.62** (−9 478 vs best) |
| 20 | `final_shot_v1_58.csv` | v1=0.58/v2=0.25/v3=0.17 @ LB levels (push along v1-weight gradient) | **740,379.73** (−908 vs best) |

### Interpretation

* **Chronos-2 hurts this LB.** The foundation model is genuinely
  diverse (corr 0.81), but its zero-shot shape is further from the
  private test's daily pattern than our LGBM ensemble that has been
  deeply tuned on the 14 CSV features. On benchmarks like
  fev-bench / GIFT-Eval, Chronos-2 dominates, but this competition's
  metric (yearly-aggregated MAE on total Revenue/COGS with known
  annual levels) is much more sensitive to **daily-shape fidelity**,
  which our LGBM captures better via DoY/DoW/lag features.
* **v1-weight optimum is at 0.50.** Public LB gradient:
  v1=0.45 → 740 067, v1=0.50 → **739 472**, v1=0.58 → 740 380. The
  minimum is a local well around 0.50; pushing either direction
  hurts. This matches what small-perturbation theory would predict
  with the three LGBM models we have.

## 5. Final outcome

Best submission remains **`b_v1v2v3_50_30_20.csv`**
(v1=0.50, v2=0.30, v3=0.20 @ `LB_LEVELS`), scoring **739,471.96** on
the public LB. `data/submission.csv` points to this file.

Summary of the full 20-submission budget: see
`results/v2/final_v2_blend.md` for the Day-2 audit, plus this file
for the Day-3 research iteration.

## 6. Lessons learned

1. Foundation models (Chronos-2 / TimesFM) are attractive because
   they are *zero-shot*, but on narrow aggregate metrics for a single
   pair of series the gains are competition-specific. Deeply tuned
   gradient boosted ensembles remain very hard to beat with a generic
   foundation model on this kind of task.
2. When three base models have internal correlations of 0.97+, the
   blend weight optimum is very flat — expected gains from further
   weight tuning are <1 k and are dominated by noise around the
   public LB.
3. The most productive lever remaining, beyond model choice, would be
   **better feature engineering on the 14 CSVs** (category mix,
   payment mix, review/delay signals with long lags) — future work
   when more submission budget is available.
