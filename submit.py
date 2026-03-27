"""
AllClear - Data Battle IA PAU 2026
submit.py : Génère predictions_submit.csv au format du jury
            + calcule le meilleur theta sur les données d'entraînement

Format de sortie :
  airport, airport_alert_id, prediction_date, predicted_date_end_alert, confidence
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_engine import build_features

TEST_FILE    = "test_data/dataset_set.csv"
TRAIN_FILE   = "segment_alerts_all_airports_train.csv"
MODEL_FILE   = "model.pkl"
OUTPUT_FILE  = "predictions_submit.csv"
R_ACCEPT     = 0.02
MIN_DIST     = 3

print("="*60)
print("AllClear — Génération des prédictions pour soumission")
print("="*60)

# ── Chargement modèle ─────────────────────────────────────────
print("\n[1/5] Chargement du modèle...")
bundle = pickle.load(open(MODEL_FILE, 'rb'))
model, FEATURE_COLS = bundle['model'], bundle['features']
print(f"    {len(FEATURE_COLS)} features")

# ── Chargement données test ───────────────────────────────────
print("\n[2/5] Chargement des données de test...")
df_test = pd.read_csv(TEST_FILE)
df_test['date'] = pd.to_datetime(df_test['date'], utc=True)
df_test = df_test[df_test['icloud'] == False].copy()

# Les éclairs avec airport_alert_id = données fournies (début d'alerte)
known = df_test[df_test['airport_alert_id'].notna()].copy()
known['airport_alert_id'] = known['airport_alert_id'].astype(int)
print(f"    {len(known):,} éclairs CG | {known.groupby(['airport','airport_alert_id']).ngroups} alertes")

# ── Feature engineering ───────────────────────────────────────
print("\n[3/5] Calcul des features...")
known = known.sort_values(['airport','airport_alert_id','date']).reset_index(drop=True)
# is_last_lightning_cloud_ground est False pour tous (données tronquées)
known['is_last_lightning_cloud_ground'] = 0
known_feat = build_features(known)

X = known_feat[FEATURE_COLS].fillna(0)
known_feat['prob'] = model.predict_proba(X)[:, 1]
print(f"    Probabilités calculées.")

# ── Génération des prédictions au format jury ─────────────────
print("\n[4/5] Formatage des prédictions...")
# Pour chaque éclair :
# - prediction_date = date de l'éclair (quand on émet la prédiction)
# - predicted_date_end_alert = date de l'éclair lui-même
#   (on dit : "à ce moment, je pense que l'alerte est terminée")
# - confidence = probabilité du modèle
# Le jury prendra le premier éclair où confidence >= theta
# et vérifiera si des éclairs dangereux tombent après cette date

rows = []
for (airport, alert_id), grp in known_feat.groupby(['airport', 'airport_alert_id']):
    grp = grp.sort_values('date')
    for _, row in grp.iterrows():
        rows.append({
            'airport':                   airport,
            'airport_alert_id':          alert_id,
            'prediction_date':           row['date'],
            'predicted_date_end_alert':  row['date'],  # on dit "l'alerte finit maintenant"
            'confidence':                row['prob'],
        })

predictions = pd.DataFrame(rows)
# Tronquer à la seconde — supprimer les nanosecondes ajoutées pour dédupliquer
predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"]).dt.floor("s")
predictions["predicted_date_end_alert"] = pd.to_datetime(predictions["predicted_date_end_alert"]).dt.floor("s")
predictions.to_csv(OUTPUT_FILE, index=False)
print(f"    {len(predictions):,} prédictions → {OUTPUT_FILE}")

# ── Calcul du meilleur theta sur données train ────────────────
print("\n[5/5] Calcul du meilleur theta (données d'entraînement)...")
df_train = pd.read_csv(TRAIN_FILE)
df_train['date'] = pd.to_datetime(df_train['date'], utc=True)
df_train = df_train[df_train['icloud'] == False].copy()

tot_l3 = len(df_train[df_train['dist'] < MIN_DIST])
alerts_train = df_train.groupby(['airport','airport_alert_id'])

# Charger predictions.csv (généré par train.py) et évaluer
pred_df = pd.read_csv('predictions.csv')
pred_df['date'] = pd.to_datetime(pred_df['date'], format='ISO8601')

thetas = [i/20 for i in range(1, 20)]
results = {}
for theta in thetas:
    over = pred_df[pred_df['prob'] >= theta]
    if len(over) == 0:
        continue
    over_min = over.groupby(['airport','airport_alert_id'])['date'].min()
    gain, missed = 0, 0
    for (airport, alert_id), end_pred in over_min.items():
        try:
            lts = alerts_train.get_group((airport, alert_id))
            end_base = pd.to_datetime(lts['date'], utc=True).max() + pd.Timedelta(minutes=30)
            gain += (end_base - end_pred).total_seconds()
            dangerous = pd.to_datetime(lts[lts['dist'] < MIN_DIST]['date'], utc=True)
            missed += (dangerous > end_pred).sum()
        except KeyError:
            continue
    results[theta] = (gain, missed)

print(f"\n    {'Theta':>6} | {'Gain (h)':>10} | {'Risque R':>10} | {'Safe':>6}")
print("    " + "-"*42)
best_theta, best_gain = None, -1
for theta, (gain, missed) in sorted(results.items()):
    R = missed / tot_l3 if tot_l3 > 0 else 0
    safe = "✅" if R < R_ACCEPT else "❌"
    print(f"    {theta:>6.2f} | {gain/3600:>10.1f} | {R:>10.4f} | {safe}")
    if R < R_ACCEPT and gain > best_gain:
        best_gain = gain
        best_theta = theta

print(f"\n    ➤ Meilleur theta : {best_theta}")
print(f"    ➤ Gain de temps  : {best_gain/3600:.1f} heures")
print(f"    ➤ Risque R       : {results[best_theta][1]/tot_l3:.4f} < {R_ACCEPT}")

print(f"\n{'='*60}")
print(f"✅ Soumission prête : {OUTPUT_FILE}")
print(f"   Envoie ce fichier + theta={best_theta} à contact@iapau.org")
print(f"{'='*60}")
