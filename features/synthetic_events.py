# features/synthetic_events.py

import numpy as np
import pandas as pd

def inject_synthetic_crash(df, date=None, duration=3, magnitude=-0.12, price_col="close"):
    df = df.copy()
    df["synthetic_crash"] = False
    if price_col not in df.columns or "date" not in df.columns:
        raise ValueError("Input DataFrame must contain 'date' and '{}' columns".format(price_col))
    if len(df) < duration + 1:
        return df  # Not enough bars for event
    idx = None
    if date is not None:
        if date in df["date"].values:
            idx = df.index[df["date"] == date][0]
    else:
        idx = np.random.randint(len(df) - duration)
    if idx is not None:
        for i in range(duration):
            crash_idx = idx + i
            if crash_idx < len(df):
                df.at[crash_idx, "synthetic_crash"] = True
                if i == 0:
                    prev_price = df.at[crash_idx-1, price_col] if crash_idx > 0 else df.at[crash_idx, price_col]
                else:
                    prev_price = df.at[crash_idx-1, price_col]
                df.at[crash_idx, price_col] = prev_price * (1 + magnitude)
    return df

def inject_synthetic_vix_spike(df, date=None, duration=3, magnitude=0.80, vix_col="iv"):
    df = df.copy()
    df["synthetic_vix"] = False
    if vix_col not in df.columns or "date" not in df.columns:
        raise ValueError("Input DataFrame must contain 'date' and '{}' columns".format(vix_col))
    if len(df) < duration + 1:
        return df  # Not enough bars for event
    idx = None
    if date is not None:
        if date in df["date"].values:
            idx = df.index[df["date"] == date][0]
    else:
        idx = np.random.randint(len(df) - duration)
    if idx is not None:
        for i in range(duration):
            vix_idx = idx + i
            if vix_idx < len(df):
                df.at[vix_idx, "synthetic_vix"] = True
                base_iv = df.at[vix_idx, vix_col]
                df.at[vix_idx, vix_col] = base_iv * (1 + magnitude)
    return df

def inject_random_events(df, n_crashes=1, n_vix_spikes=1, crash_params=None, vix_params=None):
    df = df.copy()
    for _ in range(n_crashes):
        params = crash_params or {}
        df = inject_synthetic_crash(df, date=None, **params)
    for _ in range(n_vix_spikes):
        params = vix_params or {}
        df = inject_synthetic_vix_spike(df, date=None, **params)
    return df

def tag_event_windows(df):
    df = df.copy()
    # Defensive: Only tag if the columns exist
    crash_col = "synthetic_crash" if "synthetic_crash" in df.columns else None
    vix_col = "synthetic_vix" if "synthetic_vix" in df.columns else None
    cols = [c for c in [crash_col, vix_col] if c]
    if cols:
        df["is_event_window"] = df[cols].any(axis=1)
    else:
        df["is_event_window"] = False
    return df

def visualize_events(df, price_col="close", iv_col="iv"):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df[price_col].values, label="Price", color="blue")
    if iv_col in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df[iv_col].values, label="IV", color="orange", alpha=0.6)
        ax2.set_ylabel("IV")
    crash_idx = df.index[df.get("synthetic_crash", pd.Series([False]*len(df))).values]
    vix_idx = df.index[df.get("synthetic_vix", pd.Series([False]*len(df))).values]
    ax1.scatter(crash_idx, df[price_col].iloc[crash_idx], color="red", label="Crash", zorder=5)
    ax1.scatter(vix_idx, df[price_col].iloc[vix_idx], color="purple", label="VIX Spike", zorder=5)
    ax1.set_title("Synthetic Events in Price/IV")
    ax1.legend()
    plt.show()


