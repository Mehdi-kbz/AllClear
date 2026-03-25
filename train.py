"""
AllClear - Data Battle IA PAU 2026
train.py : Entraînement complet — XGBoost + calibration isotonique
"""
import pandas as pd
import numpy as np
import pickle, json, warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
from feature_engine import build_features

CSV_FILE     = "segment_alerts_all_airports_train.csv"
MODEL_FILE   = "model.pkl"
PRED_FILE    = "predictions.csv"
METRICS_FILE = "metrics.json"

FEATURE_COLS = [
    'elapsed_min','inter_time_min','amplitude_abs','dist','maxis',
    'cumcount',
    'count_5min','count_10min','count_15min','count_30min',
    'amp_mean_5min','amp_mean_10min','amp_mean_15min','amp_mean_30min',
    'amp_std_5min','amp_std_10min',
    'dist_mean_5min','dist_mean_10min','dist_mean_15min',
    'inter_mean_5min','inter_mean_10min',
    'close_count_5min','close_count_10min','close_count_15min','close_count_30min',
    'amp_trend','dist_trend','intensity_ratio','dist_ratio',
    'hour_sin','hour_cos','month_sin','month_cos',
    'is_close','recent_close','airport_enc',
]
TARGET = 'is_last_lightning_cloud_ground'

print("="*60)
print("AllClear — Entraînement du modèle")
print("="*60)

# 1. Chargement
print("\n[1/6] Chargement des données...")
df = pd.read_csv(CSV_FILE)
df['date'] = pd.to_datetime(df['date'], utc=True)
df = df.sort_values(['airport','airport_alert_id','date']).reset_index(drop=True)
df = df[df['icloud'] == False].copy()
print(f"    {len(df):,} éclairs CG | {df.groupby(['airport','airport_alert_id']).ngroups} alertes | {df['airport'].nunique()} aéroports")

# 2. Features
print("\n[2/6] Ingénierie des features...")
df = build_features(df)
print("    Features OK.")

df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df_clean   = df.dropna(subset=FEATURE_COLS+[TARGET]).copy()
df_clean[TARGET] = df_clean[TARGET].astype(int)
X      = df_clean[FEATURE_COLS].fillna(0)
y      = df_clean[TARGET]
groups = df_clean['alert_uid']  # grouper par alerte unique
print(f"    {len(FEATURE_COLS)} features | {y.sum()} positifs / {len(y)} observations")

# 3. Split par alerte unique
print("\n[3/6] Split train/test...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

# 4. Modèle
print("\n[4/6] Entraînement XGBoost + calibration isotonique...")
scale_pos = (y_train==0).sum()/(y_train==1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric='logloss', random_state=42, n_jobs=-1,
)
calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)
print("    Modèle entraîné et calibré.")

# 5. Évaluation
print("\n[5/6] Évaluation...")
y_prob = calibrated.predict_proba(X_test)[:,1]
auc    = roc_auc_score(y_test, y_prob)
brier  = brier_score_loss(y_test, y_prob)
print(f"    AUC-ROC : {auc:.4f} | Brier : {brier:.4f}")

metrics = {"global":{"auc":round(auc,4),"brier":round(brier,4)},"airports":{}}
test_df = df_clean.iloc[test_idx].copy()
test_df['prob']   = y_prob
test_df['target'] = y_test.values

for airport in sorted(test_df['airport'].unique()):
    sub = test_df[test_df['airport']==airport]
    if sub['target'].sum() > 0:
        a = roc_auc_score(sub['target'],sub['prob'])
        b = brier_score_loss(sub['target'],sub['prob'])
        metrics["airports"][airport] = {"auc":round(a,4),"brier":round(b,4)}
        print(f"      {airport:10s} AUC:{a:.4f} Brier:{b:.4f}")

# Minutes économisées — avec elapsed_min relatif correct
threshold  = 0.3
# elapsed_min relatif recalculé depuis la date — 100% fiable
test_df['elapsed_rel'] = test_df.groupby('alert_uid')['date'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 60
)
true_end   = test_df[test_df['target']==1].groupby('alert_uid')['elapsed_rel'].max()
early_end  = test_df[test_df['prob']>=threshold].groupby('alert_uid')['elapsed_rel'].min()
common     = true_end.index.intersection(early_end.index)
savings_s  = (true_end[common] + 30) - early_end[common]
savings    = savings_s[savings_s > 0].median()
metrics["minutes_saved"]      = round(float(savings), 1)
metrics["baseline_minutes"]   = 30
metrics["threshold"]          = threshold
print(f"    Minutes économisées (médiane) : {savings:.1f} min/alerte")

# 6. Sauvegarde
print("\n[6/6] Sauvegarde...")
with open(MODEL_FILE,'wb') as f:
    pickle.dump({'model':calibrated,'features':FEATURE_COLS}, f)
with open(METRICS_FILE,'w') as f:
    json.dump(metrics, f, indent=2)

df_clean['prob'] = calibrated.predict_proba(df_clean[FEATURE_COLS].fillna(0))[:,1]
cols = ['lightning_id','airport_alert_id','alert_uid','airport','date','lon','lat','amplitude','dist','elapsed_min','is_last_lightning_cloud_ground','prob']
df_clean[[c for c in cols if c in df_clean.columns]].to_csv(PRED_FILE, index=False)

print(f"\n{'='*60}")
print("✅ Terminé !")
print(f"   model.pkl + predictions.csv + metrics.json générés")
print(f"{'='*60}")
