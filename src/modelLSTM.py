import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler

# Assuming 'combined_data_cleaned' is the processed dataset (DataFrame)

# Identify numeric columns (this includes all engineered features)
numeric_columns = combined_data_cleaned.select_dtypes(include=[np.number]).columns

# Step 1: Prepare the data for LSTM with new features
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])  # features (exclude target)
        y.append(data[i+sequence_length, -1])    # target variable (PM2.5)
    return np.array(X), np.array(y)

sequence_length = 7  # Use the last 7 days to predict the next day
X, y = create_sequences(combined_data_cleaned[numeric_columns].values, sequence_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalize the data
scaler = StandardScaler()

# X_train and X_test are 3D arrays (samples, time steps, features) for LSTM, scaling should be applied per feature
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  # Flatten the time steps for scaling
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)  # Reshape back to 3D
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)  # Reshape back to 3D

# Step 2: Build the LSTM Model (for hyperparameter tuning)
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                   return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output the prediction (PM2.5 value)
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize Keras Tuner for LSTM tuning
tuner = kt.Hyperband(build_lstm_model, objective='val_loss', max_epochs=10, factor=3, directory='my_dir', project_name='lstm_tuning')

# Perform the tuning
tuner.search(X_train_scaled, y_train, epochs=10, validation_data=(X_test_scaled, y_test))

# Best hyperparameters for LSTM
best_lstm_hp = tuner.get_best_hyperparameters()[0]
print("Best LSTM hyperparameters:", best_lstm_hp.values)

# Step 3: Train the Best LSTM Model with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

lstm_model = build_lstm_model(best_lstm_hp)
lstm_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# Step 4: Get LSTM Predictions (or hidden states) to use as features for XGBoost
lstm_preds_train = lstm_model.predict(X_train_scaled)
lstm_preds_test = lstm_model.predict(X_test_scaled)

# Save LSTM predictions for next part (XGBoost)
np.save('lstm_preds_train.npy', lstm_preds_train)
np.save('lstm_preds_test.npy', lstm_preds_test)

# You can also save the model if you need to
lstm_model.save('best_lstm_model.h5')
