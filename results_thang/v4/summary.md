# EDA v4 — Datathon 2026 · The Gridbreakers

> **Deliverable for Task 2** (60 pts). Report built on the official
> **NeurIPS 2025 LaTeX template**, framed as a *Revenue System Diagnosis*:
> every finding is a symptom → mechanism → forecasting implication → quantified
> action, designed to maximize scores on the official rubric
> (Viz 15 / Depth 25 / Business 15 / Storytelling 5).
>
> **Page budget.** Per NeurIPS rules, **the main content is ≤ 4 pages**;
> **references, appendix, and the NeurIPS Paper Checklist do not count** against
> the limit and follow on pages 5 onward.

## Files

| File | Purpose |
| --- | --- |
| `main.pdf` | **Main submission** — NeurIPS 2025, 4-page main content + references + 3 appendices + NeurIPS checklist. |
| `main_vietnam.pdf` | Vietnamese reading version (same figures/layout as `main.pdf`). Not intended as the graded submission. |
| `main.tex` | LaTeX source for the report. |
| `main_vietnam.tex` | Vietnamese LaTeX source (pdfLaTeX + T5 + babel). |
| `neurips_2025.sty` | Official NeurIPS 2025 style file (copied from `report/` so `pdflatex` finds it). |
| `report/neurips_2025.{pdf,tex,sty}` | Pristine template reference provided by the organisers. |
| `eda_v4.py` | **Canonical pipeline.** Loads 14 CSVs, builds the daily panel, runs 4 statistical tests, writes `metrics_v4.json`, and renders 5 multi-panel figures as vector PDF + PNG. Deterministic, idempotent. |
| `eda_v4.ipynb` | Executed notebook wrapping `eda_v4.py` for reproducibility. |
| `eda_v4.html` | HTML export of the notebook. |
| `metrics_v4.json` | Every number quoted in the report. No hand-typed values. |
| `images/fig{1..5}_*.pdf/.png` | Publication-quality multi-panel figures. |
| `Makefile` | `make` to rebuild everything and verify the main-content page count. |
| `report.tex`, `report.pdf` | Earlier twocolumn (non-NeurIPS) draft kept as a historical reference. The graded deliverable is `main.pdf`. |

## How to build

```bash
cd results/v4

# (1) Regenerate metrics + figures
uv run python eda_v4.py

# (2) Re-execute the notebook and export HTML (optional)
uv run jupyter nbconvert --to notebook --execute eda_v4.ipynb --output eda_v4.ipynb
uv run jupyter nbconvert --to html eda_v4.ipynb

# (3) Compile the PDF (NeurIPS template)
latexmk -pdf main.tex

# Vietnamese reading version
latexmk -pdf main_vietnam.tex

# or simply:
make        # pipeline + pdf + verify
make verify # re-check that main content <= 4 pages
make verify_vi # verify the Vietnamese PDF too
```

`make verify` locates the first appendix (`app:audit`) in the LaTeX aux
file, derives the last main-content page, and asserts it is ≤ 4.

## Report structure

### Main content (pages 1–4, the graded part)

| Pg | Sections | Core content |
| --- | --- | --- |
| 1 | Title, abstract, §1 Problem framing, §2 Core findings (F1 Regime) + **Fig 1** | Three abstract hooks, data contract, regime-break diagnosis. |
| 2 | §2 cont. (F2 Seasonality) + **Fig 2**, §3 Structural (F3 Concentration) | Seasonality (May peak, Dec trough) + category concentration. |
| 3 | §3 cont. (F4 Promos, F5 Allocation, F6 Orders) + **Fig 3** | Promo paradox, inventory paradox, leading-signal ranking. |
| 4 | **Fig 4**, §4 Prescriptive synthesis + **Table 1** (levers), limitations, ack | Rolling-origin CV design, 7-row recommendations table, honest-limits paragraph. |

### Non-counting pages (5–8): references, appendix, checklist

| Pg | Content |
| --- | --- |
| 5 | References · Appendix A (data audit, FK verification, Table 3) · Appendix B (returns, customer mix) · Appendix C (**Fig 5** CV diagram, leakage blacklist **Table 2**) start |
| 6 | Appendix C cont. · Appendix D (reproducibility: code, env, commands, licensing) |
| 7–8 | NeurIPS Paper Checklist (all 15 items answered with justification) |

## Findings (quick index)

1. **Regime break in 2019**, not COVID. −40.5% daily mean, Welch *t*=28.6, *p*<10⁻¹⁶⁰, *d*=0.89. Lever: time-decay / truncate training.
2. **May is the peak, not Q4.** Grand mean deviation: May +53%, Dec −41%. Lever: shift ad budget Q4→Q2.
3. **80/15/3/2 category concentration.** HHI=0.67. Lever: hierarchical Streetwear vs rest forecast.
4. **Promos are defensive.** Heavy-promo days −13.3% median revenue (Mann-Whitney, *p*<10⁻¹²). Lever: predictive promo trigger.
5. **Inventory paradox.** 58–70% SKUs stock-out *and* 68–87% over-stock simultaneously. Lever: allocation, not volume; +10–15% ceiling.
6. **Orders (lead), not sessions**, carry the signal. Traffic gains only +0.005 ρ at lag −1. Lever: two-stage model orders × AOV.

## Rubric mapping

| Criterion | Points | Evidence |
| --- | --- | --- |
| Visualization quality | 15 | 5 multi-panel vector PDFs; Okabe-Ito palette (colorblind-safe); every panel has a title, axis, unit, legend; captions self-contained and interpretive. |
| Analytical depth | 25 | Each of 6 findings explicitly hits Descriptive → Diagnostic → Predictive → Prescriptive in a single paragraph. Four significance tests (*t*, Mann-Whitney *U*, Spearman, Cohen's *d*), lead-lag sweep for traffic, plus explicit leakage contract in Table 2. |
| Business insight | 15 | Main Table 1 gives 7 quantified actions with expected Δ (%, VND), source section, and horizon (now / 1 mo / 1 qtr / 2 qtr). Appendix Table 2 is an explicit leakage-safe feature contract. |
| Storytelling & creativity | 5 | Narrative = "Revenue System Diagnosis": 3 counter-intuitive hooks in the abstract (regime before COVID, May not Q4, defensive promos), tied to operational mechanisms. NeurIPS template projects seriousness and forces a self-audit (checklist). |

## Data reconciliation notes

- `inventory.csv` contains 5 denormalised columns in addition to those in `data/README.md`; we use only the schema-conformant ones.
- `web_traffic.csv` is missing `conversion_rate`; we derive it from `orders / sessions` where needed.
- Monthly totals reconstructed from `orders`⋈`order_items`⋈`products` match `sales.csv` within 0.3% (Appendix A).
- All monetary values in the report are VND; figures note units in the axis label or caption.
