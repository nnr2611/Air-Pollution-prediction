import numpy as np
import pandas as pd

from preprocessing import POLLUTANTS, WEATHER

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
DEFAULT_WINDOWS = (24, 168)


def add_lag_features(df, cols, lags):
    """Lagged copies of ``cols``. Rows are not dropped; callers drop NaNs once."""
    new = {f"{c}_lag{l}": df[c].shift(l) for c in cols for l in lags}
    return df.assign(**new)


def create_rolling_features(df, cols, windows):
    """Trailing rolling means, shifted by one so a row never sees its own value."""
    new = {
        f"{c}_roll{w}": df[c].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        for c in cols for w in windows
    }
    return df.assign(**new)


def add_time_features(df):
    """Calendar features; cyclical ones are encoded as sin/cos."""
    idx = df.index
    new = {
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
        "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "month_sin": np.sin(2 * np.pi * idx.month / 12),
        "month_cos": np.cos(2 * np.pi * idx.month / 12),
        "day_of_week": idx.dayofweek,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
    }
    return df.assign(**new)


def add_wind_features(df):
    rad = np.deg2rad(df["wd_deg"])
    return df.assign(wd_sin=np.sin(rad), wd_cos=np.cos(rad))


def build_features(df, horizon=24, target="PM2.5", lags=DEFAULT_LAGS, windows=DEFAULT_WINDOWS):
    """Build the supervised table for ONE station (hourly index).

    The target is ``target`` shifted ``horizon`` hours into the future, computed
    per station so values never leak across stations.
    """
    df = add_wind_features(add_time_features(df))
    lag_cols = [c for c in POLLUTANTS if c in df.columns]
    df = add_lag_features(df, lag_cols, lags)
    df = create_rolling_features(df, lag_cols + ["TEMP", "WSPM"], windows)
    df = df.assign(y=df[target].shift(-horizon))
    return df.drop(columns=["wd"], errors="ignore")


def build_dataset(clean, horizon=24, **kw):
    """Run build_features per station and stack the results."""
    parts = []
    for _, g in clean.groupby("station"):
        parts.append(build_features(g, horizon=horizon, **kw))
    out = pd.concat(parts)
    out["station"] = out["station"].astype("category")
    return out


def feature_columns(df):
    """Model inputs: everything except identifiers and the target."""
    return [c for c in df.columns if c not in ("y", "wd")]
