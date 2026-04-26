# Phase 2 Metric Audit

Independent recomputation from raw CSV files completed successfully.

| Check | Value | Expected | Status | Note |
|---|---:|---:|---|---|
| order_items rows after joins | 714669 | 714669 | PASS |  |
| missing order join rows | 0 | 0 | PASS |  |
| missing product join rows | 0 | 0 | PASS |  |
| max daily Revenue diff vs sales.csv | 1.862645149230957e-09 | 0.0 | PASS | Floating-point rounding only. |
| max daily COGS diff vs sales.csv | 0.00499926891643554 | 0.0 | PASS | Floating-point rounding only. |
| product rows exported | 1598 | 1598 | PASS |  |
| product table max diff: units | 0.0 | 0.0 | PASS |  |
| product table max diff: orders | 0.0 | 0.0 | PASS |  |
| product table max diff: revenue | 1.862645149230957e-09 | 0.0 | PASS |  |
| product table max diff: gross_margin | 1.4901161193847656e-08 | 0.0 | PASS |  |
| product table max diff: promo_revenue | 3.725290298461914e-09 | 0.0 | PASS |  |
| product table max diff: return_qty | 0.0 | 0.0 | PASS |  |
| product table max diff: refund_amount | 9.313225746154785e-10 | 0.0 | PASS |  |
| category table max diff: products | 0.0 | 0.0 | PASS |  |
| category table max diff: units | 0.0 | 0.0 | PASS |  |
| category table max diff: orders | 0.0 | 0.0 | PASS |  |
| category table max diff: revenue | 0.0 | 0.0 | PASS |  |
| category table max diff: gross_margin | 2.384185791015625e-07 | 0.0 | PASS |  |
| category table max diff: promo_revenue | 0.0 | 0.0 | PASS |  |
| category table max diff: return_qty | 0.0 | 0.0 | PASS |  |
| category table max diff: refund_amount | 0.0 | 0.0 | PASS |  |
| top category by gross margin | Streetwear | Streetwear | PASS |  |
| top category gross margin | 1738676765.0 | 1738676765.0 | PASS |  |
| top product by gross margin | SaigonFlex UM-43 | SaigonFlex UM-43 | PASS |  |
| top product gross margin | 130456863.28 | 130456863.28 | PASS |  |
| promo revenue share | 0.330814 | 0.330814 | PASS |  |
| promo line share | 0.386635 | 0.386635 | PASS |  |
| wrong_size return share | 0.349708 | 0.349708 | PASS |  |
| inventory product coverage | 1624 / 2412 |  | PASS | Inventory is monthly and does not cover every catalog product. |
| geography join missing rows | 0 | 0 | PASS |  |
