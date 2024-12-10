import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_xgboost(X_train, y_train, params=None):
    """Train an XGBoost model."""
    if params is None:
        params = {"objective": "reg:squarederror", "n_estimators": 100, "learning_rate": 0.1}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def build_lstm_model(input_shape):
    """Build and compile an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def evaluate_model(model, X_test, y_test, lstm=False):
    """Evaluate model performance."""
    y_pred = model.predict(X_test) if not lstm else model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {"MAE": mae, "MSE": mse}
