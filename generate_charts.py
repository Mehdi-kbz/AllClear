"""
AllClear - Data Battle IA PAU 2026
generate_charts.py : Génère les 5 graphes de résultats dans charts/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs('charts', exist_ok=True)

# ── Style global ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f8f8',
    'axes.grid':        True,
    'grid.color':       '#e0e0e0',
    'grid.linewidth':   0.8,
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

COLORS = {
    'Ajaccio':  '#0A84FF',
    'Bastia':   '#30d158',
    'Biarritz': '#ff9f0a',
    'Nantes':   '#ff453a',
    'Pise':     '#bf5af2',
}
SEUILS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ── Chargement ─────────────────────────────────────────────────
print("Chargement predictions.csv...")
df = pd.read_csv('predictions.csv')
df['date'] = pd.to_datetime(df['date'], format='ISO8601')
df['alert_uid'] = df['airport'] + '_' + df['airport_alert_id'].astype(str)
df['elapsed_rel'] = df.groupby('alert_uid')['date'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 60
)
df['target'] = df['is_last_lightning_cloud_ground'].fillna(0).astype(int)
multi = df.groupby('alert_uid').filter(lambda x: len(x) > 1)
last  = multi[multi['target']==1].groupby('alert_uid')['elapsed_rel'].max()

# ══════════════════════════════════════════════════════════════
# 1. COURBE ROC par aéroport
# ══════════════════════════════════════════════════════════════
print("Graphe 1 — Courbe ROC...")
fig, ax = plt.subplots(figsize=(7, 6))

for airport, color in COLORS.items():
    sub = df[df['airport'] == airport]
    if sub['target'].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(sub['target'], sub['prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{airport} (AUC = {roc_auc:.3f})')

ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Aléatoire')
ax.set_xlabel('Taux de faux positifs')
ax.set_ylabel('Taux de vrais positifs')
ax.set_title('Courbe ROC — par aéroport')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig('charts/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("   charts/roc_curves.png ✓")

# ══════════════════════════════════════════════════════════════
# 2. CALIBRATION — modèle calibré + courbe idéale
# ══════════════════════════════════════════════════════════════
print("Graphe 2 — Calibration...")

fig, ax = plt.subplots(figsize=(7, 6))

# Global
frac_pos, mean_pred = calibration_curve(df['target'], df['prob'], n_bins=10, strategy='quantile')
ax.plot(mean_pred, frac_pos, marker='o', color='#0A84FF', lw=2.5, label='Modèle calibré (global)')

# Par aéroport
for airport, color in COLORS.items():
    sub = df[df['airport'] == airport]
    if sub['target'].sum() < 5:
        continue
    try:
        fp, mp = calibration_curve(sub['target'], sub['prob'], n_bins=8, strategy='quantile')
        ax.plot(mp, fp, marker='.', color=color, lw=1.5, alpha=0.7, ls='--', label=airport)
    except Exception:
        pass

ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Calibration parfaite')
ax.set_xlabel('Probabilité prédite moyenne')
ax.set_ylabel('Fraction de positifs réels')
ax.set_title('Calibration des probabilités après isotonique')
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('charts/calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("   charts/calibration.png ✓")

# ══════════════════════════════════════════════════════════════
# 3. DISTRIBUTION des probs — dernier éclair vs autres
# ══════════════════════════════════════════════════════════════
print("Graphe 3 — Distribution des probabilités...")
fig, ax = plt.subplots(figsize=(7, 5))

others = df[df['target']==0]['prob']
lasts  = df[df['target']==1]['prob']

bins = np.linspace(0, 1, 40)
ax.hist(others, bins=bins, alpha=0.6, color='#636366', label='Éclairs non-finaux', density=True)
ax.hist(lasts,  bins=bins, alpha=0.75, color='#0A84FF', label='Dernier éclair de l\'alerte', density=True)

ax.axvline(lasts.median(), color='#0A84FF', lw=1.5, ls='--',
           label=f'Médiane dernier éclair ({lasts.median():.2f})')
ax.set_xlabel('Probabilité prédite P(fin d\'alerte)')
ax.set_ylabel('Densité')
ax.set_title('Distribution des probabilités prédites')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('charts/prob_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   charts/prob_distribution.png ✓")

# ══════════════════════════════════════════════════════════════
# 4. COUVERTURE vs SÉCURITÉ par seuil
# ══════════════════════════════════════════════════════════════
print("Graphe 4 — Couverture vs Sécurité...")
coverages, safes = [], []
for t in SEUILS:
    early   = multi[multi['prob']>=t].groupby('alert_uid')['elapsed_rel'].min()
    common  = last.index.intersection(early.index)
    safe    = (early[common] >= last[common]).mean() * 100
    covered = len(common) / len(last) * 100
    coverages.append(covered)
    safes.append(safe)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(coverages, safes, c=SEUILS, cmap='RdYlGn', s=120, zorder=5,
                vmin=0.3, vmax=0.9, edgecolors='white', linewidths=1.5)
ax.plot(coverages, safes, color='#aeaeb2', lw=1.5, zorder=4)

for i, t in enumerate(SEUILS):
    ax.annotate(f'  seuil {t}', (coverages[i], safes[i]),
                fontsize=9, color='#333')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Seuil de probabilité', fontsize=10)

ax.set_xlabel('Couverture — % alertes où le modèle se prononce')
ax.set_ylabel('Sécurité — % décisions prises après le dernier éclair')
ax.set_title('Arbitrage Couverture / Sécurité par seuil')
ax.set_xlim(0, 55); ax.set_ylim(88, 101)
plt.tight_layout()
plt.savefig('charts/coverage_safety.png', dpi=150, bbox_inches='tight')
plt.close()
print("   charts/coverage_safety.png ✓")

# ══════════════════════════════════════════════════════════════
# 5. FAUX NÉGATIFS vs seuil
# ══════════════════════════════════════════════════════════════
print("Graphe 5 — Risque de faux négatifs...")
fn_rates, coverages2 = [], []
for t in SEUILS:
    early   = multi[multi['prob']>=t].groupby('alert_uid')['elapsed_rel'].min()
    common  = last.index.intersection(early.index)
    fn_rate = (early[common] < last[common]).mean() * 100
    covered = len(common) / len(last) * 100
    fn_rates.append(fn_rate)
    coverages2.append(covered)

fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

ax1.plot(SEUILS, fn_rates,   color='#ff453a', lw=2.5, marker='o', ms=7, label='% faux négatifs')
ax2.plot(SEUILS, coverages2, color='#0A84FF', lw=2.5, marker='s', ms=7, ls='--', label='% alertes couvertes')

ax1.set_xlabel('Seuil de probabilité')
ax1.set_ylabel('% faux négatifs (risque)', color='#ff453a')
ax2.set_ylabel('% alertes couvertes', color='#0A84FF')
ax1.tick_params(axis='y', colors='#ff453a')
ax2.tick_params(axis='y', colors='#0A84FF')
ax1.set_title('Risque de faux négatifs et couverture selon le seuil')

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, fontsize=10, loc='center right')
plt.tight_layout()
plt.savefig('charts/false_negatives.png', dpi=150, bbox_inches='tight')
plt.close()
print("   charts/false_negatives.png ✓")

print("\n Tous les graphes générés dans charts/")
