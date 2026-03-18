# DATA BATTLE 2026 — Prédiction de fin d'orage
### Meteorage × IA Pau | Équipe AllClear

---

## Ce que fait ce projet

Les aéroports ont besoin de savoir **quand un orage est vraiment terminé** pour reprendre les opérations au sol. Aujourd'hui, la règle est simple : attendre 30 minutes après le dernier éclair. Le problème : beaucoup d'orages se terminent bien avant — et chaque minute perdue coûte de l'argent.

Ce pipeline entraîne un modèle de machine learning qui **estime en temps réel la probabilité que l'éclair courant soit le dernier** dans une zone d'alerte (rayon de 20 km autour d'un aéroport). Quand cette probabilité dépasse un seuil, l'aéroport peut lever l'alerte plus tôt en toute sécurité.

---

## Structure du projet

```
databattle2026_pipeline.py   ← Code principal (à exécuter)
requirements.txt             ← Packages Python à installer
README.md                    ← Ce fichier

outputs/
  fig1_model_comparison.png  ← Comparaison des performances des modèles
  fig2_detailed_eval.png     ← ROC, Précision-Rappel, Calibration
  fig3_alert_simulation.png  ← Simulation sur une vraie alerte
```

---

## Prérequis

### 1. Python 3.10+
Vérifier avec :
```bash
python --version
```
Téléchargeable sur https://www.python.org si nécessaire.

### 2. Le fichier de données
Placer `segment_alerts_all_airports_train.csv` dans le **même dossier** que `databattle2026_pipeline.py`.

### 3. Packages Python
Tout installer en une commande :
```bash
pip install -r requirements.txt
```

Installe :
- **pandas / numpy** — manipulation des données
- **scikit-learn** — split, métriques, régression logistique, calibration
- **xgboost** — modèle principal (arbres de décision boostés)
- **matplotlib** — graphiques
- **pyarrow** — sauvegarde rapide au format parquet

---

## Lancement

```bash
# Se placer dans le dossier du projet
cd chemin/vers/le/projet

# Lancer le pipeline complet
python databattle2026_pipeline.py
```

Le script va :
1. Charger et nettoyer le CSV (~1 sec)
2. Calculer toutes les features (~3 min — les fenêtres glissantes sont lentes)
3. Entraîner tous les modèles (~2 min)
4. Sauvegarder 3 figures dans ./outputs/

---

## Fonctionnement du code (étape par étape)

### ÉTAPE 1 — Chargement des données

Le CSV brut contient une ligne par éclair. On ne conserve que les éclairs appartenant à une **alerte** (airport_alert_id non nul). Cela filtre de 507 071 à 56 599 lignes.

La **variable cible** est `is_last_lightning_cloud_ground` :
- `True`  = cet éclair était le dernier nuage-sol de son alerte
- `False` = d'autres éclairs vont suivre

Il s'agit d'un problème de **classification binaire** avec un fort déséquilibre de classes : seulement 4,6 % des éclairs sont positifs (derniers de leur alerte).

---

### ÉTAPE 2 — Feature engineering

C'est l'étape la plus importante. Les données brutes contiennent seulement : date, position, amplitude, distance. Il faut donner au modèle du **contexte** : l'orage est-il encore actif ? S'affaiblit-il ? Y a-t-il eu un long silence ?

**A) Position temporelle**
- `elapsed_min` — depuis combien de temps cette alerte dure-t-elle ?
- `rank_in_alert` — est-ce le 1er, 5e ou 50e éclair ?

**B) Temps inter-éclairs** ← feature la plus importante
- `inter_time_min` — temps depuis l'éclair précédent
- Intuition : un long silence = l'orage est peut-être en train de se terminer

**C) Statistiques glissantes (4 fenêtres : 5 / 10 / 15 / 30 min)**
Pour chaque éclair, on regarde les N dernières minutes et on calcule :
- `count_Xm` — combien d'éclairs ont eu lieu ? (intensité de l'orage)
- `amp_mean_Xm`, `amp_max_Xm` — statistiques d'amplitude (niveau d'énergie)
- `dist_mean_Xm`, `dist_min_Xm` — statistiques de distance (proximité)

**D) Features de tendance**
- `amp_trend_3` — l'amplitude augmente-t-elle ou diminue-t-elle sur les 3 derniers éclairs ?
- `dist_trend_3` — l'orage se rapproche-t-il ou s'éloigne-t-il ?

**E) Ratios d'intensité**
- `intensity_ratio_5_15` = count_5m / count_15m
- Proche de 1,0 = orage encore actif récemment
- Proche de 0,0 = orage en train de mourir (activité récente bien inférieure au passé)

**F) Features spatiales**
- `az_sin`, `az_cos` — direction de l'éclair (azimut encodé de façon circulaire)
- `amp_x_dist` — éclairs puissants proches de l'aéroport = risque élevé

**G) Temps cyclique**
- `month_sin/cos`, `hour_sin/cos` — encodés pour que décembre soit "proche" de janvier

**H) Identité de l'aéroport**
- `airport_enc` — entier 0-4, permet au modèle d'apprendre des patterns spécifiques à chaque aéroport

> Pourquoi encoder le mois en sin/cos ?
> Avec mois=12 et mois=1, le modèle penserait qu'ils sont éloignés.
> sin/cos boucle : cos(360°) = cos(0°), donc janvier et décembre sont voisins.

---

### ÉTAPE 3 — Découpage train/test

On découpe par **ID d'alerte**, pas par ligne. C'est critique.

**Mauvaise méthode (découpage par ligne) :** Le modèle pourrait voir l'éclair n°5 de l'alerte n°42 à l'entraînement et l'éclair n°8 de la même alerte au test. Comme les éclairs d'un même orage sont corrélés, cela gonfle artificiellement les métriques.

**Bonne méthode (découpage par groupe) :** Tous les éclairs de l'alerte n°42 vont entièrement en train OU entièrement en test. Le modèle est évalué sur des orages qu'il n'a jamais vus.

On utilise 80 % des alertes pour l'entraînement, 20 % pour le test.

---

### ÉTAPE 4 — Entraînement des modèles

Quatre modèles sont comparés :

| Modèle | Fonctionnement | Pourquoi on l'essaie |
|--------|---------------|----------------------|
| **Heuristique** | `prob = inter_time / 30` | Reproduit l'approche du timer fixe actuel |
| **Régression logistique** | Combinaison linéaire des features | Baseline simple et interprétable |
| **Random Forest** | Ensemble d'arbres de décision | Capture les non-linéarités |
| **XGBoost** | Arbres boostés par gradient | Modèle le plus performant, gère le déséquilibre |

XGBoost utilise `scale_pos_weight = 19` (ratio négatifs/positifs) pour compenser le déséquilibre de classes. Sans cela, le modèle prédirait "pas le dernier" en permanence et aurait raison 95 % du temps — ce qui est inutile.

---

### ÉTAPE 5 — Calibration

Les probabilités brutes de XGBoost sont souvent trop confiantes. On applique une **calibration isotonique** (`CalibratedClassifierCV`) qui ajuste ces probabilités pour qu'elles correspondent à la réalité observée : si le modèle prédit 70 %, l'alerte doit effectivement se terminer dans 70 % des cas similaires.

C'est ce score calibré qui est exposé comme **indicateur de confiance** à l'utilisateur.

---

### ÉTAPE 6 — Évaluation

Quatre métriques sont reportées :

- **AUC-ROC** (plus haut = mieux, max 1,0)
  Capacité du modèle à classer les positifs au-dessus des négatifs.
  0,5 = aléatoire, 1,0 = parfait.

- **AUC-PR** (plus haut = mieux)
  Aire sous la courbe Précision-Rappel. Plus informative que la ROC sur données déséquilibrées.
  Un classifieur aléatoire obtient AUC-PR ≈ 0,046 (= taux de base).

- **Brier score** (plus bas = mieux)
  Erreur quadratique moyenne des probabilités prédites vs. vraies étiquettes.
  Mesure la **calibration** autant que le classement.

- **Log-loss** (plus bas = mieux)
  Entropie croisée. Pénalise fortement les prédictions confiantes et fausses.

---

## Exemple concret : ce que fait le modèle en pratique

Séquence d'événements à l'aéroport de Biarritz un après-midi d'été :

```
14:00:00 — Éclair à 18,3 km, amplitude -45 kA  → P(fin) = 0,02  (orage qui commence)
14:03:22 — Éclair à 12,1 km, amplitude -31 kA  → P(fin) = 0,01  (très actif)
14:07:55 — Éclair à  9,4 km, amplitude -67 kA  → P(fin) = 0,01  (puissant, proche)
14:09:10 — Éclair à 11,2 km, amplitude -22 kA  → P(fin) = 0,03  (légère baisse)
14:14:03 — Éclair à 16,8 km, amplitude -12 kA  → P(fin) = 0,11  (long silence, loin, faible)
14:21:47 — Éclair à 19,1 km, amplitude  -8 kA  → P(fin) = 0,38  (7 min de silence, très faible, loin)
14:29:15 — Éclair à 18,6 km, amplitude  -5 kA  → P(fin) = 0,72  ← LE MODÈLE SE DÉCLENCHE
                                                                    "Fin d'alerte probable"
```

À 14:29:15 :
- Plus de 7 minutes depuis le dernier éclair
- L'amplitude est passée de 67 kA à 5 kA (très faible)
- L'orage s'éloigne (19 km, en bordure de zone)
- Activité sur les 5 dernières minutes : 1 éclair (contre 4 il y a 30 min)

L'ancien système attendrait jusqu'à **14:59:15** (30 min après cet éclair) pour lever l'alerte. Le modèle reconnaît que l'orage se termine et suggère de la lever **maintenant**.

**Temps gagné : ~30 minutes.**

---

## Résultats

| Modèle | AUC-ROC | AUC-PR | Brier |
|--------|---------|--------|-------|
| Heuristique (timer) | 0,606 | 0,164 | 0,039 |
| Régression logistique | 0,903 | 0,298 | 0,114 |
| Random Forest | 0,953 | 0,505 | 0,063 |
| **XGBoost (calibré)** | **0,959** | **0,513** | **0,042** |

Le modèle XGBoost identifie correctement le dernier éclair avec un AUC-ROC de 0,959 — il classe le vrai dernier éclair au-dessus de 95,9 % des éclairs non-finaux en moyenne.

---

## Pistes d'amélioration

1. **Modèle de survie** (Cox / Weibull) — reformuler le problème comme "temps jusqu'à la fin de l'orage" plutôt que classification. Plus solide théoriquement.

2. **Modèles par aéroport** — Pise et Biarritz ont des profils d'orage très différents. Des modèles séparés pourraient améliorer les performances.

3. **Optimisation du seuil** — choisir le seuil de probabilité qui minimise le temps perdu pour chaque aéroport (dépend du coût des fausses alarmes vs. des retards inutiles). Ce choix est une décision opérationnelle et RSE, pas purement technique.

4. **Validation croisée temporelle** — entraîner sur 2016-2020, tester sur 2021-2022 pour simuler un vrai déploiement en production.
