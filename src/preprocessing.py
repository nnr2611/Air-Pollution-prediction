from pathlib import Path

import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def load_raw(data_dir="data/raw"):
    """Load every PRSA station CSV into one frame indexed by (station, datetime)."""
    files = sorted(Path(data_dir).glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No PRSA_Data_*.csv files found in {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


def clean_station(df, max_gap_hours=6):
    """Clean one station: dedupe, regular hourly grid, short-gap interpolation.

    Gaps longer than ``max_gap_hours`` stay NaN rather than being invented.
    """
    df = df.drop_duplicates(subset="datetime").set_index("datetime").sort_index()
    df = df.asfreq("h")
    num_cols = [c for c in POLLUTANTS + WEATHER if c in df.columns]
    df[num_cols] = df[num_cols].interpolate(method="time", limit=max_gap_hours)
    df["station"] = df["station"].ffill().bfill()
    df["wd"] = df["wd"].ffill(limit=max_gap_hours)
    # Wind direction as a number so models can use it (sin/cos added in features).
    df["wd_deg"] = df["wd"].map(WIND_DEGREES)
    return df


def clean_data(df, max_gap_hours=6):
    """Apply clean_station to each station of a load_raw() frame."""
    parts = [clean_station(g, max_gap_hours) for _, g in df.groupby("station")]
    return pd.concat(parts)


def split_by_time(df, test_fraction=0.2):
    """Chronological split on the datetime index, shared by all stations."""
    cutoff = df.index.sort_values().unique()[int(df.index.nunique() * (1 - test_fraction))]
    return df[df.index < cutoff], df[df.index >= cutoff]
