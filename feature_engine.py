import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy().reset_index(drop=True)

    # Clé unique par alerte
    df['alert_uid'] = df['airport'] + '_' + df['airport_alert_id'].astype(str)

    # Ajouter un nanoseconde offset pour les timestamps dupliqués
    df['date'] = df['date'] + pd.to_timedelta(df.groupby(['alert_uid','date']).cumcount(), unit='ns')

    g = df.groupby('alert_uid')

    df['amplitude_abs']   = df['amplitude'].abs()
    df['elapsed_sec']     = g['date'].transform(lambda x: (x - x.min()).dt.total_seconds())
    df['elapsed_min']     = df['elapsed_sec'] / 60.0
    df['inter_time_sec']  = g['date'].diff().dt.total_seconds().fillna(0)
    df['inter_time_min']  = df['inter_time_sec'] / 60.0
    df['cumcount']        = g.cumcount() + 1
    df['is_close_raw']    = (df['dist'] < 3).astype(float)

    for w in [5, 10, 15, 30]:
        wl    = f'{w}min'
        w_sec = w * 60
        parts = {}

        for uid, grp in df.groupby('alert_uid'):
            # set_index sur date — maintenant unique grâce au nanoseconde offset
            grp2 = grp.set_index('date').sort_index()
            r = pd.DataFrame({
                f'count_{wl}':        grp2['amplitude_abs'].rolling(f'{w_sec}s', closed='both').count(),
                f'amp_mean_{wl}':     grp2['amplitude_abs'].rolling(f'{w_sec}s', closed='both').mean(),
                f'amp_std_{wl}':      grp2['amplitude_abs'].rolling(f'{w_sec}s', closed='both').std().fillna(0),
                f'dist_mean_{wl}':    grp2['dist'].rolling(f'{w_sec}s', closed='both').mean(),
                f'inter_mean_{wl}':   grp2['inter_time_min'].rolling(f'{w_sec}s', closed='both').mean(),
                f'close_count_{wl}':  grp2['is_close_raw'].rolling(f'{w_sec}s', closed='both').sum(),
            })
            r.index = grp.index  # remettre l'index original du df
            parts[uid] = r

        combined = pd.concat(parts.values()).sort_index()
        for col in combined.columns:
            df[col] = combined[col].reindex(df.index).fillna(0)

        print(f"    Fenêtre {wl} OK")

    df['amp_trend']       = g['amplitude_abs'].transform(lambda x: x.rolling(3, min_periods=1).mean().diff().fillna(0))
    df['dist_trend']      = g['dist'].transform(lambda x: x.rolling(3, min_periods=1).mean().diff().fillna(0))
    df['intensity_ratio'] = df['amplitude_abs'] / (g['amplitude_abs'].transform('mean') + 1e-6)
    df['dist_ratio']      = df['dist']          / (g['dist'].transform('mean') + 1e-6)

    df['hour']       = df['date'].dt.hour
    df['month']      = df['date'].dt.month
    df['hour_sin']   = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)

    df['is_close']     = (df['dist'] < 3).astype(int)
    df['recent_close'] = (df['close_count_5min'] > 0).astype(int)

    airport_map = {a: i for i, a in enumerate(sorted(df['airport'].unique()))}
    df['airport_enc']    = df['airport'].map(airport_map)
    df['alert_progress'] = df['cumcount'] / (g['cumcount'].transform('max') + 1e-6)

    return df
