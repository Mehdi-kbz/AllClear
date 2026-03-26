# 🏆 Data Battle IA PAU 2026 – AllClear

## 👥 Équipe
- Nom de l'équipe : AllClear
- Membres :
  - Mehdi Khabouze
  - Hiba Knizia
  - Kenza Rzal
  - Ilhame Daoui

## 🎯 Problématique
Les aéroports appliquent une règle fixe : attendre 30 minutes après le dernier éclair détecté à moins de 6 miles avant de rouvrir les pistes. Cette règle ignore la dynamique réelle de l'orage et génère des immobilisations coûteuses (~12 000 $/minute). L'objectif est de remplacer ce délai fixe par une probabilité de fin d'alerte mise à jour en temps réel, éclair par éclair.

## 💡 Solution proposée
Un modèle XGBoost avec calibration isotonique qui, à chaque éclair nuage-sol détecté, produit P(fin d'alerte). 36 features construites par fenêtres glissantes temporelles (5/10/15/30 min), incluant un comptage dédié des éclairs à moins de 3 km (zone de danger opérationnel). À θ=0.3 : gain de 1 331 heures sur le dataset d'entraînement, risque R=1.84% < 2% (protocole Meteorage). AUC-ROC : 0.966 — Brier Score : 0.018.

## ⚙️ Stack technique
- Langages : Python 3.8+, HTML/CSS/JavaScript
- Frameworks : XGBoost, scikit-learn, pandas, numpy, matplotlib
- Outils : Leaflet.js, Chart.js, PapaParse
- IA (si utilisé) : XGBoost + calibration isotonique (CalibratedClassifierCV)

## 🚀 Installation & exécution

### Prérequis
- Python 3.8+
- `segment_alerts_all_airports_train.csv` dans le répertoire racine
- `test_data/dataset_set.csv` dans le sous-dossier `test_data/`

### Installation
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

### Exécution

```bash
# 1. Entraîner le modèle (génère model.pkl, predictions.csv, metrics.json)
python train.py

# 2. Générer les graphiques (génère charts/)
python generate_charts.py

# 3. Lancer l'interface web
python -m http.server 8000
# Ouvrir http://localhost:8000

# 4. Générer les prédictions pour soumission jury (données test)
python submit.py
```
