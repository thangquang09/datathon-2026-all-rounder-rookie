import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

from src.models.sales_forecasting import REPO_ROOT

fig, ax = plt.subplots(1, 1, figsize=(18, 24))
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis('off')
fig.patch.set_facecolor('white')

# ACADEMIC COLOR PALETTE — minimal, grayscale + accent
BG = 'white'
BOX_DATA = '#37474F'       # dark gray
BOX_FE = '#546E7A'         # medium gray  
BOX_PIPE_A = '#455A64'
BOX_PIPE_B = '#607D8B'
BOX_PIPE_C = '#78909C'
BOX_ENSEMBLE = '#263238'
BOX_OUTPUT = '#37474F'
ACCENT = '#1565C0'         # blue accent
SHAP_BG = '#FFF9C4'        # light yellow highlight
LIGHT_BG = '#F5F5F5'
BORDER = '#B0BEC5'
TEXT_DARK = '#212121'
TEXT_LIGHT = '#FFFFFF'
TEXT_MID = '#616161'

def rect(ax, x, y, w, h, fc='white', ec=BORDER, lw=1.0, zorder=3):
    r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(r)

def rrect(ax, x, y, w, h, fc='white', ec=BORDER, lw=1.0, rad=0.1, zorder=3):
    r = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={rad}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(r)

def txt(ax, x, y, s, fs=10, c=TEXT_DARK, w='normal', ha='center', va='center', zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, fontweight=w, zorder=zorder,
            fontfamily='serif')

def arrow(ax, x1, y1, x2, y2, c='#90A4AE', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=2)

def arrow_down(ax, x, y1, y2, **kw):
    c = kw.get('color', '#90A4AE')
    l = kw.get('lw', 1.2)
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=l), zorder=2)

# ============================================================
# TITLE
# ============================================================
txt(ax, 9, 23.4, 'Figure 1: Forecasting Pipeline Architecture', fs=14, c=TEXT_DARK, w='bold')
txt(ax, 9, 23.0, 'Two-Tier Design: Feature Engineering (Tier 1) → Ensemble Blending (Tier 2)', 
    fs=9, c=TEXT_MID, w='normal')

# ============================================================
# TIER 0: DATA
# ============================================================
rect(ax, 0.5, 20.8, 17, 1.8, fc=LIGHT_BG, ec=BORDER, lw=0.8)
txt(ax, 9, 22.3, 'Input Data (13 CSV files)', fs=11, c=TEXT_DARK, w='bold')

data_labels = ['sales', 'orders', 'order_items', 'products', 'payments',
               'shipments', 'returns', 'reviews', 'web_traffic', 
               'promotions', 'inventory', 'customers', 'geography']
bw = 17 / 7 - 0.15
for i in range(7):
    x = 0.6 + i * (17/7)
    rect(ax, x, 21.0, bw, 0.8, fc=BOX_DATA, ec='none', lw=0)
    txt(ax, x + bw/2, 21.4, data_labels[i], fs=7.5, c=TEXT_LIGHT, w='normal')
for i in range(6):
    x = 0.6 + i * (17/7)
    rect(ax, x, 20.95, bw, 0.6, fc='#546E7A', ec='none', lw=0)
    txt(ax, x + bw/2, 21.15, data_labels[7+i], fs=6.5, c=TEXT_LIGHT, w='normal')

# ============================================================
# TIER 1: FEATURE ENGINEERING
# ============================================================
rect(ax, 0.5, 17.2, 17, 3.2, fc='#FAFAFA', ec=ACCENT, lw=1.5)
txt(ax, 1.2, 20.1, 'TIER 1', fs=12, c=ACCENT, w='bold', ha='left')
txt(ax, 16.8, 20.1, '221 Features', fs=9, c=TEXT_MID, ha='right', w='normal')

fe = [
    (0.8,  17.5, 2.6, 1.8, 'Calendar &\nSeasonality',   '61%',  '#2E7D32'),
    (3.6,  17.5, 2.6, 1.8, 'Target\nLags',               '24%',  '#388E3C'),
    (6.4,  17.5, 2.6, 1.8, 'Holiday &\nEvent',           '11%',  '#43A047'),
    (9.2,  17.5, 2.0, 1.8, 'Regime\nLevel',              '1.5%', '#66BB6A'),
    (11.4, 17.5, 2.0, 1.8, 'Anchor\nLevel',              '1.5%', '#81C784'),
    (13.6, 17.5, 2.0, 1.8, 'Horizon\nEffect',            '0.2%', '#A5D6A7'),
    (15.8, 17.5, 1.5, 1.8, 'Other',                      '<0.1%','#C8E6C9'),
]
for (x, y, w, h, name, pct, col) in fe:
    rect(ax, x, y, w, h, fc=col, ec='#1B5E20', lw=0.6)
    txt(ax, x+w/2, y+h/2+0.35, name, fs=9, c='white', w='bold')
    txt(ax, x+w/2, y+h/2-0.45, pct, fs=8, c='#E8F5E9', w='normal')

# Arrows: Data → FE
for i in range(5):
    x = fe[i][0] + fe[i][2]/2
    arrow_down(ax, x, 20.8, 19.3, color=BORDER, lw=0.8)

# ============================================================
# TIER 1.5: 3 PIPELINES  (side by side)
# ============================================================
arrow_down(ax, 3.5, 17.2, 16.2, color='#2E7D32', lw=1.5)
arrow_down(ax, 9.0, 17.2, 16.2, color='#2E7D32', lw=1.5)
arrow_down(ax, 14.5, 17.2, 16.2, color='#2E7D32', lw=1.5)

# --- Pipeline A ---
rect(ax, 0.5, 7.4, 5.5, 8.6, fc='white', ec=BOX_PIPE_A, lw=1.2)
txt(ax, 3.25, 15.7, 'Pipeline A', fs=11, c=BOX_PIPE_A, w='bold')
txt(ax, 3.25, 15.25, 'Recursive LightGBM', fs=8, c=TEXT_MID, w='normal')

# 9 models: 3x3 grid
for si in range(3):
    for se in range(3):
        bx = 0.8 + se * 1.7
        by = 13.5 - si * 1.4
        rect(ax, bx, by, 1.5, 1.1, fc=BOX_PIPE_A, ec='none', lw=0)
        specs = ['A1','A2','A3'][si]
        seeds = ['s42','s123','s999'][se]
        txt(ax, bx+0.75, by+0.55, f'LGBM\n{specs}|{seeds}', fs=6.5, c='white', w='normal')

# Bag label
rect(ax, 1.0, 9.4, 4.5, 0.6, fc='#263238', ec='none', lw=0)
txt(ax, 3.25, 9.7, '9-Model Bagged Ensemble', fs=8, c='white', w='bold')

# Arrows
for si in range(3):
    for se in range(3):
        bx = 0.8 + se * 1.7 + 0.75
        by = 13.5 - si * 1.4
        arrow_down(ax, bx, by, 10.0, color=BORDER, lw=0.6)

# CV weight label
rect(ax, 1.0, 7.7, 4.5, 1.3, fc=LIGHT_BG, ec=BORDER, lw=0.6)
txt(ax, 3.25, 8.6, 'CV-Weighted Average', fs=8, c=TEXT_DARK, w='bold')
txt(ax, 3.25, 8.1, 'R² ≈ 0.68 (Revenue)', fs=7.5, c=TEXT_MID, w='normal')

# --- Pipeline B ---
rect(ax, 6.3, 7.4, 5.5, 8.6, fc='white', ec=BOX_PIPE_B, lw=1.2)
txt(ax, 9.05, 15.7, 'Pipeline B', fs=11, c=BOX_PIPE_B, w='bold')
txt(ax, 9.05, 15.25, 'Direct Multi-Step', fs=8, c=TEXT_MID, w='normal')

b_items = [
    (6.6, 13.5, 4.9, 1.0, 'LightGBM (direct 548d)',   '~45% wt'),
    (6.6, 12.2, 4.9, 1.0, 'RidgeCV (log1p + scaled)', '~35% wt'),
    (6.6, 10.9, 4.9, 1.0, 'DoY Prior (seasonal)',     '~20% wt'),
]
colors_b = ['#455A64', '#607D8B', '#90A4AE']
for i, (bx, by, bw, bh, name, wt) in enumerate(b_items):
    rect(ax, bx, by, bw, bh, fc=colors_b[i], ec='none', lw=0)
    txt(ax, bx+bw/2, by+bh/2+0.15, name, fs=7.5, c='white', w='normal')
    txt(ax, bx+bw/2, by+bh/2-0.25, wt, fs=7, c='#CFD8DC', w='normal')

# Ensemble
rect(ax, 7.0, 9.4, 4.1, 0.6, fc='#37474F', ec='none', lw=0)
txt(ax, 9.05, 9.7, 'Grid-Search Weighted Ensemble', fs=8, c='white', w='bold')

for (bx, by, bw, bh, *_) in b_items:
    arrow_down(ax, bx + bw/2, by, 10.0, color=BORDER, lw=0.6)

# SHAP highlight
rect(ax, 6.6, 7.7, 4.9, 1.3, fc=SHAP_BG, ec='#F9A825', lw=1.5)
txt(ax, 9.05, 8.6, '[*] SHAP + Gain Analysis', fs=9, c='#E65100', w='bold')
txt(ax, 9.05, 8.05, 'TreeExplainer | 2000 samples', fs=7, c='#F57F17', w='normal')

arrow_down(ax, 9.05, 9.4, 9.0, color=BORDER, lw=0.8)

# R² label
rect(ax, 7.0, 7.5, 4.1, 0.0, fc='none', ec='none', lw=0)
txt(ax, 9.05, 7.55, 'R² = 0.80 (Revenue)', fs=7.5, c=TEXT_MID, w='normal')

# --- Pipeline C ---
rect(ax, 12.1, 7.4, 5.5, 8.6, fc='white', ec=BOX_PIPE_C, lw=1.2)
txt(ax, 14.85, 15.7, 'Pipeline C', fs=11, c=BOX_PIPE_C, w='bold')
txt(ax, 14.85, 15.25, 'Legacy M5-Style Blend', fs=8, c=TEXT_MID, w='normal')

c_items = [
    (12.4, 14.0, 5.0, 0.9, 'v1: No-proxy recursive LGBM',   '50%'),
    (12.4, 12.8, 5.0, 0.9, 'v2: Improved FE + calibration',  '30%'),
    (12.4, 11.6, 5.0, 0.9, 'v3: Hyperparameter tuned',       '5%'),
    (12.4, 10.4, 5.0, 0.9, 'v4: Big FE overhaul',            '15%'),
]
colors_c = ['#455A64', '#546E7A', '#78909C', '#607D8B']
for i, (bx, by, bw, bh, name, wt) in enumerate(c_items):
    rect(ax, bx, by, bw, bh, fc=colors_c[i], ec='none', lw=0)
    txt(ax, bx+2.0, by+bh/2+0.1, name, fs=7, c='white', w='normal', ha='left')
    txt(ax, bx+bw-0.4, by+bh/2+0.1, wt, fs=7.5, c='#CFD8DC', w='bold', ha='right')

# M5 blend
rect(ax, 12.8, 9.4, 4.2, 0.6, fc='#263238', ec='none', lw=0)
txt(ax, 14.85, 9.7, 'M5-Style Weighted Blend', fs=8, c='white', w='bold')

for (bx, by, bw, bh, *_) in c_items:
    arrow_down(ax, bx + bw/2, by, 10.0, color=BORDER, lw=0.6)

# Legacy note
rect(ax, 12.8, 7.7, 4.2, 1.3, fc=LIGHT_BG, ec=BORDER, lw=0.6)
txt(ax, 14.85, 8.6, 'Earlier iterations of Pipeline A', fs=7.5, c=TEXT_DARK, w='normal')
txt(ax, 14.85, 8.05, 'Each: recursive LGBM + calibration', fs=7, c=TEXT_MID, w='normal')

# Dashed arrow A → C
ax.plot([6.0, 12.1], [14.5, 14.5], '--', color=BORDER, lw=0.8, zorder=1)
ax.annotate('', xy=(12.1, 14.5), xytext=(11.5, 14.5),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=0.8), zorder=2)
txt(ax, 9.05, 14.8, 'iterative development', fs=7, c=BORDER, w='normal', ha='center')

# ============================================================
# TIER 2: FINAL BLEND
# ============================================================
rect(ax, 0.5, 3.0, 17, 4.0, fc='#FAFAFA', ec='#C62828', lw=1.5)
txt(ax, 1.2, 6.7, 'TIER 2', fs=12, c='#C62828', w='bold', ha='left')

# Calibration
rect(ax, 1.0, 4.8, 4.0, 1.4, fc='#5D4037', ec='none', lw=0)
txt(ax, 3.0, 5.7, 'Regime Recovery', fs=9.5, c='white', w='bold')
txt(ax, 3.0, 5.1, 'Yearly level calibration\n2023: +40% | 2024: +80%', fs=7.5, c='#D7CCC8', w='normal')

# Final Blend
rect(ax, 5.5, 4.8, 7.0, 1.4, fc='#C62828', ec='none', lw=0)
txt(ax, 9.0, 5.7, 'Final Blend', fs=12, c='white', w='bold')
txt(ax, 9.0, 5.05, '80% Pipeline C  +  20% Pipeline B', fs=9, c='#FFCDD2', w='normal')

# Output
rect(ax, 13.0, 4.8, 4.2, 1.4, fc=BOX_OUTPUT, ec='none', lw=0)
txt(ax, 15.1, 5.7, 'submission.csv', fs=10, c='white', w='bold')
txt(ax, 15.1, 5.1, '548 days × 2 targets\n(Revenue + COGS)', fs=7.5, c='#B0BEC5', w='normal')

# Arrows: Pipelines → Final
arrow(ax, 3.25, 7.4, 3.0, 6.2, c='#455A64', lw=1.5)
arrow(ax, 9.05, 7.4, 9.0, 6.2, c='#607D8B', lw=1.5)
arrow(ax, 14.85, 7.4, 14.85, 6.2, c='#78909C', lw=1.5)

# calibration → blend → output
arrow(ax, 5.0, 5.5, 5.5, 5.5, c='#5D4037', lw=1.2)
arrow(ax, 12.5, 5.5, 13.0, 5.5, c=BOX_OUTPUT, lw=1.2)

# ============================================================
# PERFORMANCE TABLE
# ============================================================
txt(ax, 9.0, 3.7, 'Performance Comparison (Walk-Forward CV, 548-day horizon)', 
    fs=10, c=TEXT_DARK, w='bold')

# Table header
rect(ax, 1.5, 2.8, 15, 0.5, fc=BOX_DATA, ec=BORDER, lw=0.5)
cols = [(2.5, 'Model'), (7.5, 'MAE'), (9.5, 'RMSE'), (11.5, 'R²'), (14.0, 'Note')]
for (cx, label) in cols:
    txt(ax, cx, 3.05, label, fs=8.5, c='white', w='bold')

# Table rows
rows_data = [
    ('Seasonal Naive (lag-364)', '828,618', '1,218,866', '0.38', 'baseline'),
    ('XGBoost (single model)',   '576,791', '821,306',   '0.72', 'best single model'),
    ('Recursive Ensemble (9 models)', '631,776', '879,062', '0.68', 'Pipeline A'),
    ('Direct LGBM+Ridge (3 models)',  '503,012', '733,825', '0.80', 'Pipeline B [*]'),
    ('Final Ensemble (16 models)',    '—', '—', '≈0.82', '80% C + 20% B'),
]
for i, (model, mae, rmse, r2, note) in enumerate(rows_data):
    y = 2.3 - i * 0.4
    bg = LIGHT_BG if i % 2 == 0 else 'white'
    rect(ax, 1.5, y, 15, 0.4, fc=bg, ec=BORDER, lw=0.3)
    fs = 7.5
    txt(ax, 2.5, y+0.2, model, fs=fs, c=TEXT_DARK, w='normal', ha='left')
    txt(ax, 7.5, y+0.2, mae,   fs=fs, c=TEXT_DARK, w='normal')
    txt(ax, 9.5, y+0.2, rmse,  fs=fs, c=TEXT_DARK, w='normal')
    txt(ax, 11.5, y+0.2, r2,   fs=fs, c=ACCENT, w='bold')
    txt(ax, 14.0, y+0.2, note, fs=fs, c=TEXT_MID, w='normal')

# ============================================================
# BOTTOM NOTE
# ============================================================
txt(ax, 9, 0.3, 'Key: All 16 models share identical feature groups (Tier 1). '
    'Ensemble blending (Tier 2) reduces variance without changing feature-level insights. '
    'SHAP analysis on Direct LGBM (Pipeline B) serves as the canonical explainability artifact.',
    fs=7.5, c=TEXT_MID, w='normal', ha='center')

plt.tight_layout(pad=0.3)
out_path = REPO_ROOT / "docs" / "images" / "PIPELINE_FLOWCHART.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Done!")
