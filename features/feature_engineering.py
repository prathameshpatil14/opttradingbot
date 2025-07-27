# features/feature_engineering.py

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# ----------- TECHNICAL INDICATORS -----------

def bollinger_bands(close, window=20, num_std=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower, ma

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def rsi(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

# ----------- OPTION FEATURES (IV & GREEKS) -----------

def black_scholes_greeks_vectorized(df):
    S = df["close"].values.astype(float)
    K = df["strike"].fillna(df["close"]).values.astype(float) if "strike" in df.columns else S
    T = df["days_to_expiry"].fillna(2).values.astype(float) / 252.0 if "days_to_expiry" in df.columns else np.full_like(S, 2/252.0)
    sigma = df["iv"].fillna(0.15).values.astype(float) if "iv" in df.columns else np.full_like(S, 0.15)
    r = df["risk_free_rate"].fillna(0.06).values.astype(float) if "risk_free_rate" in df.columns else np.full_like(S, 0.06)

    # Avoid invalid math (vectorized mask)
    mask = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    N = len(S)
    delta = np.zeros(N)
    gamma = np.zeros(N)
    theta = np.zeros(N)
    vega = np.zeros(N)

    # Only compute for valid rows
    idx = np.where(mask)[0]
    if len(idx) > 0:
        S_v = S[idx]
        K_v = K[idx]
        T_v = T[idx]
        sigma_v = sigma[idx]
        r_v = r[idx]

        d1 = (np.log(S_v / K_v) + (r_v + 0.5 * sigma_v ** 2) * T_v) / (sigma_v * np.sqrt(T_v))
        d2 = d1 - sigma_v * np.sqrt(T_v)
        from scipy.stats import norm
        delta[idx] = norm.cdf(d1)
        gamma[idx] = norm.pdf(d1) / (S_v * sigma_v * np.sqrt(T_v))
        theta[idx] = (-S_v * norm.pdf(d1) * sigma_v / (2 * np.sqrt(T_v))
                      - r_v * K_v * np.exp(-r_v * T_v) * norm.cdf(d2)) / 252
        vega[idx] = S_v * norm.pdf(d1) * np.sqrt(T_v) / 100

    # Return as DataFrame
    return pd.DataFrame({
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }, index=df.index)


def compute_iv_surface(df, window=10):
    iv_col = "iv"
    if iv_col not in df.columns:
        df[iv_col] = 0.15
    df["iv_atm_smooth"] = df[iv_col].rolling(window, min_periods=1).mean()
    return df

# ----------- MARKET REGIME DETECTION -----------

def volatility_regime(close, window=20):
    rolling_vol = close.pct_change().rolling(window).std()
    thresh = rolling_vol.median() * 1.5
    regime = (rolling_vol > thresh).astype(int)
    return regime

def trend_regime(close, window=20):
    ma = close.rolling(window).mean()
    diff = ma.diff()
    regime = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return regime

# ----------- SYNTHETIC EVENT INJECTION -----------

def inject_synthetic_events(df, crash_dates=None, vix_spike_dates=None):
    df["synthetic_crash"] = False
    df["synthetic_vix"] = False
    if crash_dates:
        df.loc[df["date"].isin(crash_dates), "synthetic_crash"] = True
    if vix_spike_dates:
        df.loc[df["date"].isin(vix_spike_dates), "synthetic_vix"] = True
    return df

# ----------- MAIN FEATURE ENGINEERING -----------

def compute_features(df, features_config):
    df = df.copy()
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col in df.columns:
            df[f"{col}_z"] = zscore(df[col].fillna(0))

    # Technicals with custom windows if provided
    if features_config.get("use_bollinger", True):
        boll_window = features_config.get("boll_window", 20)
        upper, lower, mband = bollinger_bands(df["close"], window=boll_window)
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_mband"] = mband
    if features_config.get("use_macd", True):
        fast = features_config.get("macd_fast", 12)
        slow = features_config.get("macd_slow", 26)
        signal = features_config.get("macd_signal", 9)
        macd_line, macd_signal, macd_hist = macd(df["close"], fast=fast, slow=slow, signal=signal)
        df["macd_line"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
    if features_config.get("use_rsi", True):
        rsi_win = features_config.get("rsi_window", 14)
        df["rsi"] = rsi(df["close"], window=rsi_win)

    if features_config.get("use_iv", True):
        iv_window = features_config.get("iv_window", 10)
        df = compute_iv_surface(df, window=iv_window)
    if features_config.get("use_greeks", True):
        greeks = black_scholes_greeks_vectorized(df)
        for col in greeks.columns:
            df[col] = greeks[col]

    if features_config.get("use_regime", True):
        regime_window = features_config.get("regime_window", 20)
        df["vol_regime"] = volatility_regime(df["close"], window=regime_window)
        df["trend_regime"] = trend_regime(df["close"], window=regime_window)

    if features_config.get("synthetic_events", False):
        crash_dates = features_config.get("crash_dates", [])
        vix_spike_dates = features_config.get("vix_spike_dates", [])
        df = inject_synthetic_events(df, crash_dates=crash_dates, vix_spike_dates=vix_spike_dates)

    # Robust fill for all NaNs (use new pandas style, then fill with zero)
    df = df.bfill().ffill().fillna(0)

    if "minutes_to_close" not in df.columns:
        df["minutes_to_close"] = 30

    drop_cols = [c for c in df.columns if c in ["date", "timestamp"]]
    feat_df = df.drop(columns=drop_cols, errors="ignore")

    # Remove any all-NaN columns just in case
    feat_df = feat_df.dropna(axis=1, how='all')

    # Debug print for NaNs
    if feat_df.isnull().values.any():
        print("NaNs remain in feat_df after feature engineering! Showing counts:")
        print(feat_df.isnull().sum())
        raise ValueError("NaNs detected in feature dataframe.")

    return feat_df
