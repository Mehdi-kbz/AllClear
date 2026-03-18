"""
=============================================================================
DATA BATTLE 2026 — Complete Pipeline
Meteorage | Storm End Prediction for Airports
=============================================================================

WHAT THIS DOES:
  Given lightning strike data around airports, this pipeline:
  1. Loads and cleans the raw CSV data
  2. Engineers 40+ features (rolling counts, inter-times, trends, etc.)
  3. Trains an XGBoost model to predict the probability that any given
     lightning strike is the LAST one in an alert (= end of storm alert)
  4. Evaluates the model and plots results

HOW TO RUN:
  python pipeline.py

REQUIREMENTS: see requirements.txt
=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
DATA_PATH   = "segment_alerts_all_airports_train.csv"
OUTPUT_DIR  = "outputs"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Airport name → integer encoding
AIRPORT_MAP = {'Ajaccio': 0, 'Bastia': 1, 'Biarritz': 2, 'Nantes': 3, 'Pise': 4}

# Features used for training
FEATURE_COLS = [
    # Raw lightning features
    'amplitude', 'maxis', 'dist', 'azimuth',
    # Temporal position features
    'elapsed_min', 'inter_time_min', 'inter_time_sec', 'rank_in_alert',
    # Rolling counts and statistics — 5 min window
    'count_5m', 'amp_mean_5m', 'amp_max_5m', 'dist_mean_5m', 'dist_min_5m',
    # Rolling counts and statistics — 10 min window
    'count_10m', 'amp_mean_10m', 'amp_max_10m', 'dist_mean_10m', 'dist_min_10m',
    # Rolling counts and statistics — 15 min window
    'count_15m', 'amp_mean_15m', 'amp_max_15m', 'dist_mean_15m', 'dist_min_15m',
    # Rolling counts and statistics — 30 min window
    'count_30m', 'amp_mean_30m', 'amp_max_30m', 'dist_mean_30m', 'dist_min_30m',
    # Trend features
    'amp_trend_3', 'dist_trend_3',
    # Ratio features (recent vs. past activity)
    'intensity_ratio_5_15', 'intensity_ratio_5_30',
    # Spatial features
    'az_sin', 'az_cos', 'amp_x_dist',
    # Cyclical time features
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    # Airport identity
    'airport_enc',
]


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV and return only the lightning strikes that belong
    to an alert (i.e., airport_alert_id is not null).

    Alert definition (from Meteorage):
      - An alert STARTS when a lightning strike hits within 20 km of an airport.
      - An alert ENDS after 30 consecutive minutes with no lightning within 20 km.

    The column `is_last_lightning_cloud_ground` marks the last cloud-to-ground
    strike of each alert — this is our prediction TARGET.
    """
    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Keep only lightning within an alert
    alerts = df[df['airport_alert_id'].notna()].copy()
    alerts['airport_alert_id'] = alerts['airport_alert_id'].astype(int)
    alerts = alerts.sort_values(['airport', 'airport_alert_id', 'date']).reset_index(drop=True)

    # Binary target: 1 = last CG lightning of the alert
    alerts['target'] = (alerts['is_last_lightning_cloud_ground'].astype(str) == 'True').astype(int)

    print(f"  Total lightning strikes (in alerts): {len(alerts):,}")
    print(f"  Number of alerts:                    {alerts['airport_alert_id'].nunique():,}")
    print(f"  Airports:                            {sorted(alerts['airport'].unique())}")
    print(f"  Date range:                          {alerts['date'].min().date()} → {alerts['date'].max().date()}")
    print(f"  Positive targets (last CG):          {alerts['target'].sum():,}  ({alerts['target'].mean()*100:.1f}%)")
    print()

    return alerts


# =============================================================================
# STEP 2 — FEATURE ENGINEERING
# =============================================================================

def _rolling_stats(group: pd.DataFrame, window_sec: int) -> pd.DataFrame:
    """
    For each lightning strike in `group` (one alert), compute statistics
    over all strikes that occurred within the last `window_sec` seconds.

    This is a LOOK-BACK window — we never use future data.
    For strike at time T, we aggregate strikes in (T - window, T].
    """
    group = group.sort_values('date')
    # Convert timestamps to unix seconds for fast arithmetic
    dates = group['date'].values.astype('int64') // 1_000_000_000  # nanoseconds → seconds
    amps  = group['amplitude'].abs().values
    dists = group['dist'].values

    counts, amp_means, amp_maxs, dist_means, dist_mins = [], [], [], [], []

    for i in range(len(group)):
        t    = dates[i]
        mask = (dates <= t) & (dates > t - window_sec)
        n    = mask.sum()
        counts.append(n)
        amp_means.append(amps[mask].mean()  if n > 0 else 0.0)
        amp_maxs.append(amps[mask].max()   if n > 0 else 0.0)
        dist_means.append(dists[mask].mean() if n > 0 else 0.0)
        dist_mins.append(dists[mask].min()  if n > 0 else 0.0)

    return pd.DataFrame(
        {'count': counts, 'amp_mean': amp_means, 'amp_max': amp_maxs,
         'dist_mean': dist_means, 'dist_min': dist_mins},
        index=group.index
    )


def _linear_trend(values: np.ndarray, window: int = 3) -> list:
    """
    Compute the slope of a linear fit over the last `window` values.
    Returns 0.0 for the first (window-1) elements.

    Positive slope = increasing, negative = decreasing.
    Used to detect whether amplitude or distance is growing/shrinking.
    """
    trends = []
    for i in range(len(values)):
        if i < window - 1:
            trends.append(0.0)
        else:
            segment = values[i - window + 1: i + 1]
            slope   = float(np.polyfit(range(window), segment, 1)[0])
            trends.append(slope)
    return trends


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from raw alert data. All features respect
    temporal causality (no future data leakage).

    Feature groups:
      A) Temporal position       — where are we in the alert timeline?
      B) Inter-strike time       — how long since the last strike?
      C) Rolling statistics      — how active was the storm in the last N minutes?
      D) Trend features          — is amplitude/distance increasing or decreasing?
      E) Intensity ratios        — is recent activity a large/small fraction of total?
      F) Spatial features        — direction and weighted proximity of the strike
      G) Cyclical time features  — month and hour encoded as sin/cos
      H) Airport encoding        — which airport is this?
    """
    print("=" * 60)
    print("STEP 2 — Feature Engineering")
    print("=" * 60)

    # ── A) Temporal position ──────────────────────────────────────────────
    df['alert_start'] = df.groupby('airport_alert_id')['date'].transform('min')
    df['elapsed_sec'] = (df['date'] - df['alert_start']).dt.total_seconds()
    df['elapsed_min'] = df['elapsed_sec'] / 60
    # Rank of this strike within its alert (1 = first, N = last)
    df['rank_in_alert'] = df.groupby('airport_alert_id').cumcount() + 1

    # ── B) Inter-strike time ───────────────────────────────────────────────
    # Time since the PREVIOUS strike in the same alert
    df['inter_time_sec'] = (
        df.groupby('airport_alert_id')['date']
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
    df['inter_time_min'] = df['inter_time_sec'] / 60

    # ── C) Rolling statistics (4 window sizes) ─────────────────────────────
    for w_min, label in [(5, '5m'), (10, '10m'), (15, '15m'), (30, '30m')]:
        print(f"  Computing rolling window {label}...")
        rolled = df.groupby('airport_alert_id', group_keys=False).apply(
            lambda g: _rolling_stats(g, w_min * 60)
        )
        df[f'count_{label}']     = rolled['count']
        df[f'amp_mean_{label}']  = rolled['amp_mean']
        df[f'amp_max_{label}']   = rolled['amp_max']
        df[f'dist_mean_{label}'] = rolled['dist_mean']
        df[f'dist_min_{label}']  = rolled['dist_min']

    # ── D) Trend features ─────────────────────────────────────────────────
    # Slope of |amplitude| over the last 3 strikes: positive = growing storm
    df['amp_trend_3'] = df.groupby('airport_alert_id', group_keys=False).apply(
        lambda g: pd.Series(
            _linear_trend(g.sort_values('date')['amplitude'].abs().values),
            index=g.sort_values('date').index
        )
    )
    # Slope of distance over last 3 strikes: positive = moving AWAY from airport
    df['dist_trend_3'] = df.groupby('airport_alert_id', group_keys=False).apply(
        lambda g: pd.Series(
            _linear_trend(g.sort_values('date')['dist'].values),
            index=g.sort_values('date').index
        )
    )

    # ── E) Intensity ratios ────────────────────────────────────────────────
    # A ratio close to 1.0 = recent activity ≈ past activity (stable storm)
    # A ratio close to 0.0 = storm is dying down
    df['intensity_ratio_5_15'] = df['count_5m']  / (df['count_15m'] + 1e-6)
    df['intensity_ratio_5_30'] = df['count_5m']  / (df['count_30m'] + 1e-6)

    # ── F) Spatial features ────────────────────────────────────────────────
    # Azimuth as unit vector components (avoid 0°/360° discontinuity)
    df['az_sin'] = np.sin(np.radians(df['azimuth']))
    df['az_cos'] = np.cos(np.radians(df['azimuth']))
    # Powerful strikes close to airport = high value
    df['amp_x_dist'] = df['amplitude'].abs() * (1 / (df['dist'] + 1))

    # ── G) Cyclical time features ──────────────────────────────────────────
    # Encode month and hour cyclically so January is close to December, etc.
    df['month']     = df['date'].dt.month
    df['hour']      = df['date'].dt.hour
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)

    # ── H) Airport encoding ────────────────────────────────────────────────
    df['airport_enc'] = df['airport'].map(AIRPORT_MAP)

    print(f"  Done. Dataset shape: {df.shape}  ({len(FEATURE_COLS)} features built)")
    print()
    return df


# =============================================================================
# STEP 3 — TRAIN / TEST SPLIT
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Split data into train/test sets.

    CRITICAL: We split by ALERT ID (GroupShuffleSplit), NOT by row.
    If we split randomly by row, the model would see other strikes from
    the same alert during training AND testing — that's data leakage.
    Splitting by alert ensures the model generalises to unseen storms.

    Returns: X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    print("=" * 60)
    print("STEP 3 — Train/Test Split")
    print("=" * 60)

    X      = df[FEATURE_COLS].fillna(0).values
    y      = df['target'].values
    groups = df['airport_alert_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  Train: {len(X_train):,} strikes | {y_train.sum():,} positives ({y_train.mean()*100:.1f}%)")
    print(f"  Test:  {len(X_test):,}  strikes | {y_test.sum():,}  positives ({y_test.mean()*100:.1f}%)")
    print()

    return X_train, X_test, y_train, y_test, train_idx, test_idx


# =============================================================================
# STEP 4 — MODEL TRAINING
# =============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train three models and compare them:

    1. HEURISTIC BASELINE
       Rule: probability ∝ inter-strike time (longer gap → more likely the last).
       This is essentially what airports do today (fixed 30-min timer).
       We use it as a lower bound — any real model should beat this.

    2. LOGISTIC REGRESSION
       A linear model. Good at capturing simple monotonic relationships
       (e.g., "longer inter-time = higher probability"). Needs feature scaling.
       It shows us how much value the non-linear models add.

    3. XGBOOST (main model)
       Gradient-boosted trees. Handles non-linear interactions naturally
       (e.g., "inter-time is predictive, but only if amplitude is also low").
       `scale_pos_weight` compensates for the class imbalance (95% negatives).

    WHY XGBoost WINS:
      - The storm end signal is non-linear: a long gap AND low amplitude
        AND storm moving away = very high probability of end.
      - Trees capture these conjunctions naturally.
      - It also handles missing features gracefully.

    Metrics used:
      - AUC-ROC:   overall ranking quality (threshold-free)
      - AUC-PR:    precision-recall trade-off (better for imbalanced classes)
      - Brier score: mean squared error of probabilities (lower = better)
      - Log-loss:  cross-entropy (lower = better calibration)
    """
    print("=" * 60)
    print("STEP 4 — Model Training")
    print("=" * 60)

    results  = {}
    models   = {}

    # ── Heuristic baseline ────────────────────────────────────────────────
    # Not a real model — we re-derive inter_time from the test indices
    # (passed in via X_test[:, FEATURE_COLS.index('inter_time_min')])
    inter_idx        = FEATURE_COLS.index('inter_time_min')
    heuristic_prob   = np.clip(X_test[:, inter_idx] / 30.0, 0, 1)
    results['Heuristic (inter-time)'] = _evaluate(y_test, heuristic_prob)

    # ── Logistic Regression ───────────────────────────────────────────────
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)
    lr         = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1)
    lr.fit(X_train_s, y_train)
    lr_prob    = lr.predict_proba(X_test_s)[:, 1]
    results['Logistic Regression'] = _evaluate(y_test, lr_prob)
    models['lr'] = (lr, scaler)

    # ── XGBoost ───────────────────────────────────────────────────────────
    scale_pos  = (y_train == 0).sum() / (y_train == 1).sum()
    xgb        = XGBClassifier(
        n_estimators    = 400,
        max_depth       = 6,
        learning_rate   = 0.05,
        scale_pos_weight= scale_pos,   # compensate class imbalance
        subsample       = 0.8,         # row subsampling → reduces overfitting
        colsample_bytree= 0.8,         # feature subsampling → reduces overfitting
        eval_metric     = 'logloss',
        random_state    = RANDOM_SEED,
        n_jobs          = -1,
        verbosity       = 0,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    xgb_prob   = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = _evaluate(y_test, xgb_prob)
    models['xgb'] = xgb

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n  {'Model':<30} {'AUC-ROC':>8} {'AUC-PR':>8} {'Brier':>8} {'LogLoss':>9}")
    print(f"  {'-'*65}")
    for name, m in results.items():
        print(f"  {name:<30} {m['AUC-ROC']:>8.4f} {m['AUC-PR']:>8.4f} {m['Brier']:>8.4f} {m['LogLoss']:>9.4f}")
    print()

    return xgb, xgb_prob, results


def _evaluate(y_true, y_prob):
    """Compute all evaluation metrics for a set of predictions."""
    return {
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'AUC-PR' : average_precision_score(y_true, y_prob),
        'Brier'  : brier_score_loss(y_true, y_prob),
        'LogLoss': log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
    }


# =============================================================================
# STEP 5 — EVALUATION & PLOTTING
# =============================================================================

def evaluate_and_plot(df, xgb, xgb_prob, results, y_test, test_idx):
    """
    Generate three figures:
      1. Model comparison + feature importance
      2. Per-airport AUC + calibration
      3. Real alert simulation (timeline of predicted probability)
    """
    print("=" * 60)
    print("STEP 5 — Evaluation & Plotting")
    print("=" * 60)

    BG     = '#f8f9fa'
    TEAL   = '#2a9d8f'
    CORAL  = '#e76f51'
    PURPLE = '#8338ec'
    AP_COL = {'Ajaccio': '#e76f51', 'Bastia': '#2a9d8f', 'Biarritz': '#457b9d',
              'Nantes': '#e9c46a', 'Pise': '#8338ec'}

    # Annotate test set with predictions
    df_test = df.iloc[test_idx].copy()
    df_test['xgb_prob'] = xgb_prob
    df_test['target']   = y_test

    # ── Figure 1: Model Comparison & Feature Importance ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.suptitle('XGBoost — Model Comparison & Feature Importance',
                 fontsize=15, fontweight='bold', color='#1d3557')

    # Bar chart: AUC-ROC vs AUC-PR
    ax = axes[0]
    model_names = list(results.keys())
    aucs = [results[m]['AUC-ROC'] for m in model_names]
    prs  = [results[m]['AUC-PR']  for m in model_names]
    x    = np.arange(len(model_names))
    w    = 0.35
    b1   = ax.bar(x - w/2, aucs, w, label='AUC-ROC', color=TEAL,  alpha=0.85, edgecolor='white')
    b2   = ax.bar(x + w/2, prs,  w, label='AUC-PR',  color=CORAL, alpha=0.85, edgecolor='white')
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title('Model Comparison', fontweight='bold')
    ax.legend(); ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)

    # Feature importance (top 15)
    ax = axes[1]
    imp     = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
    colors  = [PURPLE if any(k in f for k in ['count', 'inter']) else
               TEAL   if 'dist'  in f else CORAL for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors, edgecolor='white', height=0.7)
    ax.set_title('Top 15 Feature Importances (XGBoost)', fontweight='bold')
    ax.set_xlabel('Importance'); ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'fig1_model_comparison.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")

    # ── Figure 2: ROC, PR, Calibration, Per-airport AUC ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=BG)
    fig.suptitle('XGBoost — Detailed Evaluation', fontsize=15, fontweight='bold', color='#1d3557')

    # ROC curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, xgb_prob)
    ax.plot(fpr, tpr, color=PURPLE, lw=2.5, label=f'XGBoost (AUC={results["XGBoost"]["AUC-ROC"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color=PURPLE)
    ax.set_title('ROC Curve'); ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate'); ax.legend()
    ax.set_facecolor(BG); ax.spines[['top', 'right']].set_visible(False)

    # Precision-Recall curve
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_test, xgb_prob)
    ax.plot(rec, prec, color=TEAL, lw=2.5, label=f'XGBoost (AUC-PR={results["XGBoost"]["AUC-PR"]:.3f})')
    ax.axhline(y_test.mean(), color='gray', ls='--', lw=1.5, label=f'No-skill ({y_test.mean():.3f})')
    ax.fill_between(rec, prec, alpha=0.1, color=TEAL)
    ax.set_title('Precision-Recall Curve'); ax.set_xlabel('Recall')
    ax.set_ylabel('Precision'); ax.legend()
    ax.set_facecolor(BG); ax.spines[['top', 'right']].set_visible(False)

    # Calibration plot
    ax = axes[1, 0]
    frac_pos, mean_pred = calibration_curve(y_test, xgb_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, 's-', color=CORAL, lw=2, ms=7, label='XGBoost')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.set_title('Probability Calibration'); ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives'); ax.legend()
    ax.set_facecolor(BG); ax.spines[['top', 'right']].set_visible(False)

    # Per-airport AUC
    ax = axes[1, 1]
    airports = sorted(df_test['airport'].unique())
    for i, ap in enumerate(airports):
        sub     = df_test[df_test['airport'] == ap]
        auc_ap  = roc_auc_score(sub['target'], sub['xgb_prob']) if sub['target'].sum() > 0 else 0
        ax.bar(i, auc_ap, color=AP_COL.get(ap, 'gray'), edgecolor='white')
        ax.text(i, auc_ap + 0.003, f'{auc_ap:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(airports))); ax.set_xticklabels(airports, fontsize=9)
    ax.set_ylim(0.7, 1.01); ax.set_title('AUC-ROC per Airport')
    ax.set_ylabel('AUC-ROC'); ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'fig2_detailed_eval.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")

    # ── Figure 3: Real alert timeline simulation ──────────────────────────
    # Pick the alert in the test set with the most strikes
    best_alert = (
        df_test[df_test['target'] == 1]
        .groupby('airport_alert_id').size().idxmax()
    )
    sample = df_test[df_test['airport_alert_id'] == best_alert].sort_values('elapsed_min')
    last_t = sample[sample['target'] == 1]['elapsed_min'].values[-1]
    ap_name= sample['airport'].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.fill_between(sample['elapsed_min'], sample['xgb_prob'], alpha=0.25, color=PURPLE)
    ax.plot(sample['elapsed_min'], sample['xgb_prob'], color=PURPLE, lw=2.5,
            marker='o', ms=4, label='P(end of alert) — XGBoost')
    ax.axvline(last_t, color=CORAL, lw=2.5, ls='--', label=f'True last CG strike ({last_t:.0f} min)')
    ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.6, label='Decision threshold 0.5')
    ax.set_title(f'Real Alert Simulation — {ap_name} (alert #{int(best_alert)}, {len(sample)} strikes)',
                 fontweight='bold')
    ax.set_xlabel('Elapsed time since alert start (minutes)')
    ax.set_ylabel('Predicted probability of alert end')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=10)
    ax.set_facecolor(BG); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    path3 = os.path.join(OUTPUT_DIR, 'fig3_alert_simulation.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path3}")
    print()


# =============================================================================
# STEP 6 — PREDICT ON NEW DATA
# =============================================================================

def predict_alert(model, raw_alert_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of raw lightning strikes for ONE alert,
    return the same DataFrame with a new column `p_end_of_alert`.

    This is the function you would call in production:
      - Connect to Meteorage live feed
      - For each new strike, append it to `raw_alert_df`
      - Call predict_alert() → read the last row's `p_end_of_alert`
      - If p_end_of_alert > threshold → the system recommends ending the alert

    Parameters:
        model         : trained XGBClassifier
        raw_alert_df  : DataFrame with columns matching the original CSV
                        (date, lon, lat, amplitude, maxis, icloud, dist, azimuth, airport)

    Returns:
        DataFrame with all original columns + `p_end_of_alert`
    """
    df = raw_alert_df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').reset_index(drop=True)

    # Assign a dummy alert id so existing functions work
    df['airport_alert_id'] = 0
    df['target']           = 0  # unknown for new data

    df = engineer_features(df)

    X = df[FEATURE_COLS].fillna(0).values
    df['p_end_of_alert'] = model.predict_proba(X)[:, 1]

    return df[['date', 'dist', 'amplitude', 'elapsed_min', 'inter_time_min',
               'count_5m', 'count_10m', 'p_end_of_alert']]


# =============================================================================
# MAIN
# =============================================================================

def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Features
    df = engineer_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(df)

    # 4. Train
    xgb_model, xgb_prob, results = train_models(X_train, X_test, y_train, y_test)

    # 5. Evaluate
    evaluate_and_plot(df, xgb_model, xgb_prob, results, y_test, test_idx)

    print("=" * 60)
    print("ALL DONE!")
    print(f"  Figures saved in: ./{OUTPUT_DIR}/")
    print("=" * 60)

    return xgb_model, df


if __name__ == '__main__':
    model, data = main()
