import pandas as pd

def add_lag_features(df, cols, lags):
    """Add lagged features for time-series modeling."""
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.dropna()  # Drop rows with NaN due to lagging
    return df

def add_time_features(df):
    """Add time-based features such as hour, day, and month."""
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df

def create_rolling_features(df, cols, windows):
    """Add rolling average features."""
    for col in cols:
        for window in windows:
            df[f"{col}_rolling{window}"] = df[col].rolling(window=window).mean()
    return df.dropna()
