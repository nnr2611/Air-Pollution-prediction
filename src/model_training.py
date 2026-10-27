import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "early_stopping_rounds": 30,
    "n_jobs": -1,
    "random_state": 42,
}


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost with early stopping on a chronologically later validation set."""
    model = xgb.XGBRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(y_true, y_pred):
    """MAE, RMSE, R2 and sMAPE (robust to near-zero PM2.5 values)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))) * 100
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "sMAPE_%": float(smape),
    }


def top_features(model, columns, n=15):
    imp = pd.Series(model.feature_importances_, index=columns)
    return imp.sort_values(ascending=False).head(n)


# ---------------------------------------------------------------- LSTM ----

LSTM_FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "wd_sin", "wd_cos", "hour_sin", "hour_cos", "month_sin", "month_cos",
]


def make_windows(data, seq_len=24, features=LSTM_FEATURES):
    """Sliding windows of the last ``seq_len`` hours, one per row of ``data``.

    ``data`` must be the per-station hourly frames stacked (as build_dataset returns),
    so windows never cross stations. Returns (X, valid) aligned with ``data`` rows;
    the first ``seq_len - 1`` rows of each station are invalid.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n, f = len(data), len(features)
    X = np.zeros((n, seq_len, f), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    stations = data["station"].to_numpy()
    for s in pd.unique(stations):
        pos = np.flatnonzero(stations == s)
        v = data.iloc[pos][features].to_numpy(dtype=np.float32)
        w = sliding_window_view(v, seq_len, axis=0).transpose(0, 2, 1)  # (T-L+1, L, F)
        X[pos[seq_len - 1:]] = w
        valid[pos[seq_len - 1:]] = True
    return X, valid


def fit_window_scaler(X_train):
    """Per-feature mean/std from training windows (NaN-aware)."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean, std = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
    return mean, np.where(std == 0, 1, std)


def apply_window_scaler(X, mean, std):
    return np.nan_to_num((X - mean) / std, nan=0.0).astype(np.float32)


def build_lstm_model(input_shape, n_stations, units=16, dropout=0.5, lr=5e-4, l2=1e-3, noise=0.2):
    """Sequence branch (LSTM) + station embedding, regressing a standardised target."""
    from keras import layers, models, optimizers, regularizers

    seq = layers.Input(shape=input_shape, name="seq")
    sid = layers.Input(shape=(), dtype="int32", name="station")
    reg = regularizers.l2(l2)
    h = layers.GaussianNoise(noise)(seq)  # input noise: discourages memorising specific episodes
    h = layers.LSTM(units, kernel_regularizer=reg, recurrent_dropout=0.0)(h)
    h = layers.Dropout(dropout)(h)
    emb = layers.Flatten()(layers.Embedding(n_stations, 4)(sid))
    h = layers.Concatenate()([h, emb])
    h = layers.Dropout(dropout)(layers.Dense(16, activation="relu", kernel_regularizer=reg)(h))
    model = models.Model([seq, sid], layers.Dense(1)(h))
    model.compile(optimizer=optimizers.Adam(lr), loss="huber")
    return model


def train_lstm(X_train, s_train, y_train, X_val, s_val, y_val, n_stations, epochs=30, batch_size=512):
    """Train on standardised log1p targets (damps PM2.5 spikes); returns (model, y_mean, y_std)."""
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    ly = np.log1p(np.clip(y_train, 0, None))
    ym, ys = float(ly.mean()), float(ly.std())
    model = build_lstm_model(X_train.shape[1:], n_stations)
    model.fit(
        [X_train, s_train], (ly - ym) / ys,
        validation_data=([X_val, s_val], (np.log1p(np.clip(y_val, 0, None)) - ym) / ys),
        epochs=epochs, batch_size=batch_size, verbose=2,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True), ReduceLROnPlateau(patience=2)],
    )
    return model, ym, ys


def predict_lstm(model, X, sid, ym, ys, batch_size=2048):
    z = model.predict([X, sid], batch_size=batch_size, verbose=0).ravel() * ys + ym
    return np.expm1(z)


def blend_weight(y_val, p_a, p_b):
    """Weight w in [0,1] minimising val MSE of w*p_a + (1-w)*p_b."""
    ws = np.linspace(0, 1, 21)
    errs = [np.mean((y_val - (w * p_a + (1 - w) * p_b)) ** 2) for w in ws]
    return float(ws[int(np.argmin(errs))])
