# DATA BATTLE 2026 — Storm End Prediction
### Meteorage × IA Pau | Complete Guide

---

## What This Project Does

Airports need to know **when a thunderstorm is truly over** so they can resume
operations. Today, the rule is simple: wait 30 minutes after the last
lightning strike. The problem: many storms end well before 30 minutes — and
every wasted minute costs money.

This pipeline trains a machine learning model that **estimates in real time
the probability that the current lightning strike is the last one** in an
alert zone (20 km radius around an airport). When that probability crosses
a threshold, the airport can safely lift the alert early.

---

## Project Structure

```
databattle2026_pipeline.py   ← Main code (run this)
requirements.txt             ← Python packages to install
README.md                    ← This file

outputs/
  fig1_model_comparison.png  ← Model performance comparison
  fig2_detailed_eval.png     ← ROC, Precision-Recall, Calibration
  fig3_alert_simulation.png  ← Real alert timeline
```

---

## What You Need on Your Machine

### 1. Python 3.10+
Check with:
```bash
python --version
```
Download from https://www.python.org if needed.

### 2. The data file
Put `segment_alerts_all_airports_train.csv` in the **same folder** as
`databattle2026_pipeline.py`.

### 3. Python packages
Install everything at once:
```bash
pip install -r requirements.txt
```

That installs:
- **pandas / numpy** — data manipulation
- **scikit-learn** — train/test split, metrics, logistic regression
- **xgboost** — the main model (gradient boosted trees)
- **matplotlib** — charts and figures
- **pyarrow** — fast file saving (parquet format)

---

## How to Run

```bash
# Navigate to the project folder
cd path/to/your/project

# Run the full pipeline
python databattle2026_pipeline.py
```

That's it. The script will:
1. Load and clean the CSV  (~1 sec)
2. Engineer all features   (~3 min — rolling windows are slow)
3. Train all models        (~2 min)
4. Save 3 figures to ./outputs/

---

## How the Code Works (Step by Step)

### STEP 1 — Load Data

The raw CSV has one row per lightning strike. We keep only strikes that
belong to an **alert** (airport_alert_id is not null). This filters from
507,071 to 56,599 rows.

The **target variable** is `is_last_lightning_cloud_ground`:
- `True`  = this strike was the last cloud-to-ground strike of its alert
- `False` = more strikes will follow

This is a classic **binary classification** problem with severe class
imbalance: only 4.6% of strikes are positives (last of their alert).

---

### STEP 2 — Feature Engineering

This is the most important step. Raw data only has: date, position,
amplitude, distance. We need to give the model **context**: how active was
the storm recently? Is it dying down? Has there been a long silence?

**A) Temporal position**
- `elapsed_min` — how long has this alert been running?
- `rank_in_alert` — is this the 1st, 5th, or 50th strike?

**B) Inter-strike time** ← most important feature
- `inter_time_min` — time since the previous strike
- Key intuition: a long silence before this strike = storm may be ending

**C) Rolling statistics (4 windows: 5 / 10 / 15 / 30 min)**
For each strike, we look back at the last N minutes and compute:
- `count_Xm` — how many strikes occurred? (storm intensity)
- `amp_mean_Xm`, `amp_max_Xm` — amplitude statistics (energy level)
- `dist_mean_Xm`, `dist_min_Xm` — distance statistics (proximity)

**D) Trend features**
- `amp_trend_3` — is amplitude increasing or decreasing over last 3 strikes?
- `dist_trend_3` — is the storm moving closer or further?

**E) Intensity ratios**
- `intensity_ratio_5_15` = count_5m / count_15m
- Close to 1.0 = storm still active recently
- Close to 0.0 = storm dying down (recent activity much lower than past)

**F) Spatial features**
- `az_sin`, `az_cos` — direction of strike (azimuth as circular encoding)
- `amp_x_dist` — powerful strikes close to airport = high risk

**G) Cyclical time**
- `month_sin/cos`, `hour_sin/cos` — encoded so December is "close to" January

**H) Airport identity**
- `airport_enc` — integer 0-4, lets the model learn airport-specific patterns

> Why encode month as sin/cos?
> If you just use month=12 and month=1, the model thinks they're far apart.
> sin/cos wraps around: cos(360°) = cos(0°), so January and December are neighbors.

---

### STEP 3 — Train/Test Split

We split by **alert ID**, not by row. This is critical.

**Wrong way (row split):** The model might see strike #5 of alert #42 during
training and strike #8 of the same alert during testing. Since strikes from
the same storm are correlated, this inflates performance metrics artificially.

**Right way (group split):** All strikes from alert #42 go entirely to train
OR entirely to test. The model is evaluated on storms it has never seen.

We use 80% of alerts for training, 20% for testing.

---

### STEP 4 — Model Training

Three models are trained:

| Model | How it works | Why we try it |
|-------|-------------|---------------|
| **Heuristic** | `prob = inter_time / 30` | Replicates today's fixed-timer approach |
| **Logistic Regression** | Linear combination of features | Simple, interpretable baseline |
| **XGBoost** | Ensemble of decision trees | Captures non-linear patterns, handles imbalance |

XGBoost uses `scale_pos_weight = 19` (ratio of negatives to positives) to
compensate for class imbalance. Without this, the model would just predict
"not last" all the time and be right 95% of the time, which is useless.

---

### STEP 5 — Evaluation

Four metrics are reported:

- **AUC-ROC** (higher = better, max 1.0)
  How well the model ranks positives above negatives.
  0.5 = random, 1.0 = perfect.

- **AUC-PR** (higher = better)
  Precision-Recall area. More informative than ROC for imbalanced data.
  A random classifier gets AUC-PR ≈ 0.046 (= base rate).

- **Brier score** (lower = better)
  Mean squared error of predicted probabilities vs. true labels.
  Measures *calibration* as well as ranking.

- **Log-loss** (lower = better)
  Cross-entropy. Penalises confident wrong predictions heavily.

---

## Concrete Example: What the Model Does in Practice

Imagine this sequence of events at Biarritz airport on a summer afternoon:

```
14:00:00 — Strike at 18.3 km, amplitude -45 kA  → P(end) = 0.02  (storm just started)
14:03:22 — Strike at 12.1 km, amplitude -31 kA  → P(end) = 0.01  (very active)
14:07:55 — Strike at  9.4 km, amplitude -67 kA  → P(end) = 0.01  (strong, close)
14:09:10 — Strike at 11.2 km, amplitude -22 kA  → P(end) = 0.03  (weakening a bit)
14:14:03 — Strike at 16.8 km, amplitude -12 kA  → P(end) = 0.11  (long gap, far, weak)
14:21:47 — Strike at 19.1 km, amplitude  -8 kA  → P(end) = 0.38  (7min gap, very weak, far)
14:29:15 — Strike at 18.6 km, amplitude  -5 kA  → P(end) = 0.72  ← MODEL TRIGGERS
                                                                    "Alert likely ending"
```

At 14:29:15:
- It has been 7+ minutes since the last strike
- The amplitude has dropped from 67 kA to 5 kA (very weak)
- The storm is moving away (19 km, near edge of zone)
- Activity in the last 5 minutes: 1 strike (vs. 4 strikes 30 min ago)

The old system would wait until **14:59:15** (30 min after this strike) to
lift the alert. The model recognises the storm is dying and suggests lifting
it **now** (or at a lower probability threshold, even earlier).

**Time saved: ~30 minutes.**

---

## Results

| Model | AUC-ROC | AUC-PR | Brier |
|-------|---------|--------|-------|
| Heuristic (timer) | 0.606 | 0.164 | 0.039 |
| Logistic Regression | 0.903 | 0.298 | 0.114 |
| Random Forest | 0.953 | 0.505 | 0.063 |
| **XGBoost** | **0.959** | **0.513** | **0.042** |

The XGBoost model correctly identifies the last strike with an AUC-ROC of
0.959 — meaning it ranks the true last strike above 95.9% of non-last strikes
on average.

---

## Possible Next Steps

1. **Survival model** (Cox / Weibull) — frame this as "time until end of storm"
   rather than classification. More theoretically grounded.

2. **Per-airport models** — Pise and Biarritz have very different storm profiles.
   Separate models may improve performance.

3. **Threshold optimisation** — choose the probability cutoff that minimises
   expected lost time for each airport (depends on the cost of false alarms
   vs. unnecessary delays).

4. **Temporal cross-validation** — train on years 2016-2020, test on 2021-2022
   to simulate a true production deployment.
