import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(32, 26))
ax.set_xlim(0, 32)
ax.set_ylim(0, 26)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

def rounded_box(ax, x, y, w, h, fc, ec='#CFD8DC', lw=1.0, alpha=1.0, radius=0.2):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                          facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)

def label(ax, x, y, text, fontsize=11, color='#212121', weight='bold', ha='center', va='center', zorder=5, alpha=1.0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, fontweight=weight, zorder=zorder, alpha=alpha)

def arrow_down(ax, x, y1, y2, **kw):
    c = kw.get('color', '#78909C')
    l = kw.get('lw', 1.5)
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=l, mutation_scale=12), zorder=2)

def arrow_right(ax, x1, y, x2, **kw):
    c = kw.get('color', '#78909C')
    l = kw.get('lw', 1.5)
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=l, mutation_scale=12), zorder=2)

def arrow_curve(ax, x1, y1, x2, y2, **kw):
    c = kw.get('color', '#78909C')
    l = kw.get('lw', 1.5)
    r = kw.get('rad', 0.2)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=l, mutation_scale=12,
                                connectionstyle=f"arc3,rad={r}"), zorder=2)

# ============================================================
# COLOR PALETTE (clean, muted, professional)
# ============================================================
COL = {
    'data':      '#E3F2FD', 'data_box':  '#1565C0', 'data_text':'#FFFFFF',
    'fe':        '#E8F5E9', 'fe_title':  '#1B5E20',
    'seasonal':  '#2E7D32', 'lag':       '#388E3C', 'holiday':  '#43A047',
    'regime':    '#66BB6A', 'anchor':    '#81C784', 'horizon':  '#A5D6A7',
    'pipe_a':    '#FFF8E1', 'pipe_a_box':'#E65100',
    'pipe_b':    '#F3E5F5', 'pipe_b_box':'#6A1B9A',
    'pipe_c':    '#E0F2F1', 'pipe_c_box':'#00695C',
    'blend':     '#FBE9E7', 'blend_box': '#BF360C',
    'final':     '#E8EAF6', 'final_box': '#283593',
    'shap':      '#FCE4EC', 'shap_border':'#C62828',
    'perf':      '#ECEFF1',
    'arrow':     '#90A4AE',
    'section':   '#B0BEC5',
    'subtitle':  '#616161',
    'dim':       '#9E9E9E',
}

# ============================================================
# TITLE
# ============================================================
label(ax, 16, 25.3, 'FORECASTING PIPELINE — FINAL MODEL ARCHITECTURE', fontsize=18, color='#263238')
label(ax, 16, 24.8, '2-Tier Architecture: Feature Engineering (Tầng 1) → Ensemble Blending (Tầng 2)', 
      fontsize=12, color=COL['subtitle'], weight='normal')

# ============================================================
# LAYER 0: DATA SOURCES (top)
# ============================================================
rounded_box(ax, 1, 22.2, 30, 2.2, '#F5F5F5', ec='#E0E0E0', lw=1.5, radius=0.15, alpha=0.6)
label(ax, 2, 24.0, 'DATA SOURCES', fontsize=13, color='#455A64', ha='left')

data_files = ['sales.csv', 'orders.csv', 'order_items.csv', 'products.csv',
              'payments.csv', 'shipments.csv', 'returns.csv', 'reviews.csv',
              'web_traffic.csv', 'promotions.csv', 'inventory.csv', 'customers.csv', 'geography.csv']
n = len(data_files)
x_start = 2
box_w = (28) / n
for i, name in enumerate(data_files):
    x = x_start + i * box_w + 0.1
    rounded_box(ax, x, 22.5, box_w - 0.2, 1.0, COL['data_box'], ec='#0D47A1', lw=0.8, radius=0.1)
    short = name.replace('.csv','').replace('_','\n')
    ax.text(x + (box_w-0.2)/2, 23.0, short, ha='center', va='center', fontsize=6.5, 
            color='white', fontweight='normal', zorder=5, linespacing=0.85)

# ============================================================
# LAYER 1: FEATURE ENGINEERING
# ============================================================
rounded_box(ax, 1, 18.2, 30, 3.6, COL['fe'], ec='#A5D6A7', lw=1.5, radius=0.15, alpha=0.5)
label(ax, 2, 21.5, 'TANG 1: FEATURE ENGINEERING', fontsize=14, color=COL['fe_title'], ha='left')
label(ax, 30, 21.5, '221 Features', fontsize=11, color=COL['fe_title'], ha='right', weight='normal')

fe_groups = [
    (1.5,  19.0, 5.0, 2.0, 'Calendar &\nSeasonality', '61%  |  41 features', COL['seasonal']),
    (7.0,  19.0, 5.0, 2.0, 'Target\nLags',             '24%  |  26 features', COL['lag']),
    (12.5, 19.0, 5.0, 2.0, 'Holiday &\nEvent',         '11%  |  121 features', COL['holiday']),
    (18.0, 19.0, 4.5, 2.0, 'Regime\nLevel',            '1.5%  |  7 features', COL['regime']),
    (23.0, 19.0, 4.0, 2.0, 'Anchor\nLevel',            '1.5%  |  20 features', COL['anchor']),
    (27.5, 19.0, 3.0, 2.0, 'Horizon\nEffect',          '0.2%  |  5 features', COL['horizon']),
]
for (x, y, w, h, name, pct, color) in fe_groups:
    rounded_box(ax, x, y, w, h, color, ec='#1B5E20', lw=0.8, radius=0.12)
    label(ax, x + w/2, y + h/2 + 0.35, name, fontsize=10.5, color='white')
    label(ax, x + w/2, y + h/2 - 0.55, pct, fontsize=8, color='#E8F5E9', weight='normal')

# arrows: data → FE
for i in range(6):
    x = fe_groups[i][0] + fe_groups[i][2] / 2
    arrow_down(ax, x, 22.5, 21.0, color=COL['arrow'], lw=1.2)

# ============================================================
# LAYER 2: THREE PIPELINES (side by side)
# ============================================================

# --- PIPELINE A: Recursive ---
pa_x, pa_w = 1, 9.5
rounded_box(ax, pa_x, 6.0, pa_w, 11.8, COL['pipe_a'], ec='#FFCC80', lw=1.5, radius=0.15, alpha=0.45)
label(ax, pa_x + pa_w/2, 17.4, 'PIPELINE A', fontsize=13, color=COL['pipe_a_box'])
label(ax, pa_x + pa_w/2, 16.9, 'Recursive LightGBM', fontsize=10, color='#BF360C', weight='normal')

# 9 models in 3x3 grid
specs = ['spec=1', 'spec=2', 'spec=3']
seeds = ['seed=42', 'seed=123', 'seed=999']
colors_a = ['#E65100', '#EF6C00', '#F57C00']
for si, spec in enumerate(specs):
    for se, seed in enumerate(seeds):
        bx = pa_x + 0.5 + se * 3.0
        by = 14.2 - si * 1.8
        rounded_box(ax, bx, by, 2.7, 1.4, colors_a[si], ec='#BF360C', lw=0.6, radius=0.08)
        label(ax, bx + 1.35, by + 0.7, f'LGBM\n{spec}\n{seed}', fontsize=7, color='white', weight='normal')

# 9-model bag label
rounded_box(ax, pa_x + 1.5, 8.0, pa_w - 3, 1.0, '#BF360C', ec='#8B0000', lw=0.8, radius=0.1)
label(ax, pa_x + pa_w/2, 8.5, '9-Model Bag  →  CV-weighted Average', fontsize=9.5, color='white')

# Arrows from 9 models to bag
for si in range(3):
    for se in range(3):
        ax_x = pa_x + 0.5 + se * 3.0 + 1.35
        ax_y = 14.2 - si * 1.8
        ax.annotate('', xy=(ax_x, 9.0), xytext=(ax_x, ax_y),
                    arrowprops=dict(arrowstyle='-|>', color=COL['arrow'], lw=0.8, mutation_scale=10), zorder=2)

# --- PIPELINE B: Direct ---
pb_x, pb_w = 11.2, 9.5
rounded_box(ax, pb_x, 6.0, pb_w, 11.8, COL['pipe_b'], ec='#CE93D8', lw=1.5, radius=0.15, alpha=0.45)
label(ax, pb_x + pb_w/2, 17.4, 'PIPELINE B', fontsize=13, color=COL['pipe_b_box'])
label(ax, pb_x + pb_w/2, 16.9, 'Direct Multi-Step', fontsize=10, color='#4A148C', weight='normal')

b_models = [
    (pb_x + 0.5,  14.0, 2.8, 2.2, 'LightGBM',   'Direct 548-day\n~45% weight', '#7B1FA2'),
    (pb_x + 3.6,  14.0, 2.8, 2.2, 'RidgeCV',     'Log1p scaled\n~35% weight',  '#8E24AA'),
    (pb_x + 6.7,  14.0, 2.3, 2.2, 'DoY Prior',   'Seasonal base\n~20% weight', '#AB47BC'),
]
for (bx, by, bw, bh, name, desc, col) in b_models:
    rounded_box(ax, bx, by, bw, bh, col, ec='#4A148C', lw=0.8, radius=0.1)
    label(ax, bx + bw/2, by + bh/2 + 0.4, name, fontsize=11, color='white')
    label(ax, bx + bw/2, by + bh/2 - 0.4, desc, fontsize=7.5, color='#E1BEE7', weight='normal')

# Weighted ensemble
rounded_box(ax, pb_x + 1.0, 11.5, pb_w - 2.0, 1.2, '#6A1B9A', ec='#4A148C', lw=0.8, radius=0.1)
label(ax, pb_x + pb_w/2, 12.1, 'Weighted Ensemble (Grid Search)', fontsize=10, color='white')

# Arrows
for (bx, by, bw, bh, *_) in b_models:
    arrow_down(ax, bx + bw/2, by, 12.7, color=COL['arrow'], lw=1.0)

# SHAP box (highlighted)
rounded_box(ax, pb_x + 0.8, 8.5, pb_w - 1.6, 2.2, COL['shap'], ec=COL['shap_border'], lw=2.0, radius=0.12)
label(ax, pb_x + pb_w/2, 10.1, '★  SHAP + Gain Explainability', fontsize=11, color=COL['shap_border'])
label(ax, pb_x + pb_w/2, 9.4, '2000 samples  |  TreeExplainer  |  Mean |SHAP|', fontsize=8.5, color='#C62828', weight='normal')
label(ax, pb_x + pb_w/2, 8.85, '→  full_feature_importance.csv', fontsize=8, color='#E57373', weight='normal')

# Arrow: ensemble → SHAP
arrow_down(ax, pb_x + pb_w/2, 11.5, 10.7, color=COL['arrow'], lw=1.0)

# --- PIPELINE C: Legacy ---
pc_x, pc_w = 21.5, 9.5
rounded_box(ax, pc_x, 6.0, pc_w, 11.8, COL['pipe_c'], ec='#80CBC4', lw=1.5, radius=0.15, alpha=0.45)
label(ax, pc_x + pc_w/2, 17.4, 'PIPELINE C', fontsize=13, color=COL['pipe_c_box'])
label(ax, pc_x + pc_w/2, 16.9, 'Legacy Blend (M5-Style)', fontsize=10, color='#004D40', weight='normal')

c_models = [
    (pc_x + 0.5,  14.5, 4.2, 1.8, 'v1 (base FE)',          '50%', '#00838F'),
    (pc_x + 4.9,  14.5, 4.2, 1.8, 'v2 (improved FE)',      '30%', '#00897B'),
    (pc_x + 0.5,  12.3, 4.2, 1.8, 'v3 (tuned params)',     '5%',  '#00ACC1'),
    (pc_x + 4.9,  12.3, 4.2, 1.8, 'v4 (big FE overhaul)',  '15%', '#0097A7'),
]
for (bx, by, bw, bh, name, pct, col) in c_models:
    rounded_box(ax, bx, by, bw, bh, col, ec='#004D40', lw=0.8, radius=0.1)
    label(ax, bx + bw/2, by + bh/2 + 0.3, name, fontsize=10, color='white')
    label(ax, bx + bw/2, by + bh/2 - 0.35, pct, fontsize=9, color='#B2DFDB', weight='normal')

# M5 blend box
rounded_box(ax, pc_x + 1.0, 9.8, pc_w - 2.0, 1.2, '#00695C', ec='#004D40', lw=0.8, radius=0.1)
label(ax, pc_x + pc_w/2, 10.4, 'M5-Style Weighted Blend', fontsize=10, color='white')

for (bx, by, bw, bh, *_) in c_models:
    arrow_down(ax, bx + bw/2, by, 11.0, color=COL['arrow'], lw=0.8)

# dashed arrow: Pipeline A → C (legacy relationship)
ax.annotate('', xy=(pc_x + 0.3, 15.0), xytext=(pa_x + pa_w - 0.3, 15.0),
            arrowprops=dict(arrowstyle='->', color='#B0BEC5', lw=1.2,
                            connectionstyle="arc3,rad=-0.12", linestyle='dotted'), zorder=2)
label(ax, 16, 15.8, 'Legacy versions are earlier iterations of Pipeline A', 
      fontsize=7.5, color=COL['dim'], weight='normal')

# Arrows: FE → Pipelines
arrow_down(ax, pa_x + pa_w/2, 18.2, 17.8, color='#2E7D32', lw=2.0)
arrow_down(ax, pb_x + pb_w/2, 18.2, 17.8, color='#2E7D32', lw=2.0)
arrow_down(ax, pc_x + pc_w/2, 18.2, 17.8, color='#2E7D32', lw=2.0)

# ============================================================
# LAYER 3: FINAL BLEND
# ============================================================
rounded_box(ax, 1, 1.5, 30, 4.2, COL['blend'], ec='#FFAB91', lw=1.5, radius=0.15, alpha=0.4)
label(ax, 2, 5.3, 'TANG 2: ENSEMBLE & BLENDING', fontsize=14, color=COL['blend_box'], ha='left')
label(ax, 30, 5.3, 'Variance Reduction Only — No New Features', fontsize=10, color=COL['subtitle'], ha='right', weight='normal')

# Calibration
rounded_box(ax, 2, 3.3, 6.5, 1.5, '#5D4037', ec='#3E2723', lw=0.8, radius=0.1)
label(ax, 5.25, 4.3, 'Regime Recovery Calibration', fontsize=10, color='white')
label(ax, 5.25, 3.7, '2023: +40% reversion\n2024: +80% reversion', fontsize=8, color='#D7CCC8', weight='normal')

# Final Blend
rounded_box(ax, 10.5, 3.3, 11, 1.5, '#C62828', ec='#8B0000', lw=1.2, radius=0.12)
label(ax, 16, 4.35, 'FINAL BLEND', fontsize=13, color='white')
label(ax, 16, 3.65, '80% Pipeline C (M5)    +    20% Pipeline B (Direct)', fontsize=10, color='#FFCDD2', weight='normal')

# Submission
rounded_box(ax, 23.5, 3.3, 6.5, 1.5, '#1B5E20', ec='#0D3010', lw=1.0, radius=0.1)
label(ax, 26.75, 4.3, 'submission.csv', fontsize=12, color='white')
label(ax, 26.75, 3.7, '548 days × 2 targets', fontsize=9, color='#C8E6C9', weight='normal')

# Arrows: Pipelines → Final
arrow_curve(ax, pa_x + pa_w/2, 8.0, 5.25, 4.8, color='#BF360C', lw=2.0, rad=0.1)
arrow_curve(ax, pb_x + pb_w/2, 8.5, 16, 4.8, color='#6A1B9A', lw=2.0, rad=-0.05)
arrow_curve(ax, pc_x + pc_w/2, 9.8, 16, 4.8, color='#00695C', lw=2.0, rad=-0.1)

# calibration → blend
arrow_right(ax, 8.5, 4.05, 10.5, color='#5D4037', lw=1.5)

# blend → submission
arrow_right(ax, 21.5, 4.05, 23.5, color='#1B5E20', lw=2.0)

# Pipeline A goes through calibration first (via Pipeline C)
# Pipeline C already includes calibration from legacy models

# ============================================================
# PERFORMANCE BAR
# ============================================================
rounded_box(ax, 1, 1.7, 30, 1.4, '#F5F5F5', ec='#E0E0E0', lw=1.0, radius=0.1, alpha=0.7)

perf = [
    (2.5,  'Seasonal Naive',     'R² = 0.38', 0.38, '#B0BEC5'),
    (8.5,  'XGBoost (1 model)',   'R² = 0.72', 0.72, '#FFB74D'),
    (14.5, 'Direct LGBM+Ridge\n(3 models)', 'R² = 0.80', 0.80, '#CE93D8'),
    (20.5, 'Recursive Ensemble\n(9 models)', 'R² = 0.68', 0.68, '#FFCC80'),
    (26.5, 'FINAL ENSEMBLE\n(16 models)', 'R² ≈ 0.82', 0.82, '#EF5350'),
]

# Bar chart
bar_y = 2.1
bar_h = 0.25
max_w = 4.5
for (cx, name, r2_text, r2, col) in perf:
    # Name
    ax.text(cx, bar_y + bar_h + 0.55, name, ha='center', va='bottom', fontsize=8, 
            color='#455A64', fontweight='normal', linespacing=0.85)
    # Bar
    w = max_w * r2
    rounded_box(ax, cx - max_w/2, bar_y, w, bar_h, col, ec=col, lw=0, radius=0.06)
    # Background
    rounded_box(ax, cx - max_w/2, bar_y, max_w, bar_h, '#EEEEEE', ec='#E0E0E0', lw=0.5, radius=0.06, alpha=0.5)
    rounded_box(ax, cx - max_w/2, bar_y, w, bar_h, col, ec='none', lw=0, radius=0.06)
    # R2 label
    ax.text(cx - max_w/2 + w + 0.15, bar_y + bar_h/2, r2_text, ha='left', va='center', 
            fontsize=7.5, color='#455A64', fontweight='bold')

# ============================================================
# BOTTOM NOTE
# ============================================================
label(ax, 16, 1.1, 'Key Insight: All 16 models share the same features (Tầng 1) → Ensemble (Tầng 2) only reduces variance → Explaining 1 model = Explaining all', 
      fontsize=9, color='#9E9E9E', weight='normal')

plt.tight_layout(pad=0.5)
plt.savefig('/home/thangquang09/code/vinuni_hackathon/final_thang_model/PIPELINE_FLOWCHART.png', 
            dpi=180, bbox_inches='tight', facecolor='#FFFFFF')
plt.close()
print("Done!")
