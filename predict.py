"""
AllClear - Data Battle IA PAU 2026
predict.py : Prédictions rapides depuis model.pkl existant
"""
import pandas as pd, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from feature_engine import build_features

CSV_FILE   = "segment_alerts_all_airports_train.csv"
MODEL_FILE = "model.pkl"
PRED_FILE  = "predictions.csv"

print("="*60); print("AllClear — Génération des prédictions"); print("="*60)

print("\n[1/4] Chargement du modèle...")
try:
    bundle = pickle.load(open(MODEL_FILE,'rb'))
    model, FEATURE_COLS = bundle['model'], bundle['features']
    print(f"    {len(FEATURE_COLS)} features chargées.")
except FileNotFoundError:
    print("ERREUR: model.pkl introuvable. Lancez d'abord : python train.py"); exit(1)

print("\n[2/4] Chargement des données...")
df = pd.read_csv(CSV_FILE)
df['date'] = pd.to_datetime(df['date'], utc=True)
df = df.sort_values(['airport_alert_id','date']).reset_index(drop=True)
df = df[df['icloud'] == False].copy()
print(f"    {len(df):,} éclairs CG")

print("\n[3/4] Calcul des features...")
df = build_features(df)
df['is_last_lightning_cloud_ground'] = pd.to_numeric(df['is_last_lightning_cloud_ground'], errors='coerce')
df_clean = df.dropna(subset=FEATURE_COLS).copy()
df_clean['is_last_lightning_cloud_ground'] = df_clean['is_last_lightning_cloud_ground'].fillna(0).astype(int)

print("\n[4/4] Calcul des probabilités...")
df_clean['prob'] = model.predict_proba(df_clean[FEATURE_COLS].fillna(0))[:,1]
cols = ['lightning_id','airport_alert_id','airport','date','lon','lat','amplitude','dist','elapsed_min','is_last_lightning_cloud_ground','prob']
df_clean[[c for c in cols if c in df_clean.columns]].to_csv(PRED_FILE, index=False)

print(f"\n{'='*60}")
print(f" {PRED_FILE} généré — {len(df_clean):,} lignes")
print(f"   Ouvrez index.html dans vôtre navigateur.")
print(f"{'='*60}")
