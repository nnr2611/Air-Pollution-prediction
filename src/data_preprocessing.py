import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the Beijing air quality dataset."""
    return pd.read_csv(file_path, parse_dates=["time"], index_col="time")

def clean_data(df):
    """Clean data by handling missing values and duplicates."""
    df = df.drop_duplicates()
    df = df.interpolate(method='time')  # Handle missing time series data
    return df

def aggregate_regions(df, region_col="region", value_cols=None):
    """Aggregate data at the region level."""
    if value_cols is None:
        value_cols = df.select_dtypes(include=np.number).columns
    return df.groupby(region_col)[value_cols].mean().reset_index()

def split_data(df, target_col, test_size_days=7):
    """Split data into training and testing sets based on time."""
    train_data = df.iloc[:-test_size_days]
    test_data = df.iloc[-test_size_days:]
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    return X_train, X_test, y_train, y_test
