import matplotlib.pyplot as plt

# Plot actual vs. predicted values for the test set
plt.figure(figsize=(15, 6))
plt.plot(range(len(y_test)), y_test, label="Actual", color='blue', linewidth=2)
plt.plot(range(len(y_pred_xgb)), y_pred_xgb, label="Predicted", color='orange', linestyle='--')
plt.title("Actual vs Predicted Air Quality Index (Test Set)")
plt.xlabel("Time Step")
plt.ylabel("Air Quality Index")
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='green')
plt.title("Actual vs Predicted (Scatter Plot)")
plt.xlabel("Actual Air Quality Index")
plt.ylabel("Predicted Air Quality Index")
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)  # 45-degree line
plt.show()



import matplotlib.pyplot as plt
# Calculate residuals
residuals = y_test - y_pred_xgb  # No need for .values here since y_test is already a NumPy array

# Plot residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Debug: Check lengths of arrays
print(f"Length of y_test: {len(y_test)}")
print(f"Length of y_pred_xgb: {len(y_pred_xgb)}")
print(f"Length of residuals: {len(residuals)}")

# Ensure all arrays are of the same length
min_length = min(len(y_test), len(y_pred_xgb), len(residuals))

# Slice to match the smallest length
y_test = y_test[:min_length]
y_pred_xgb = y_pred_xgb[:min_length]
residuals = residuals[:min_length]

# Create a DataFrame for analysis
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred_xgb,
    "Residuals": residuals
})

# Example: Analyze mean error by temporal dimension (e.g., weekday)
# Assuming you have a 'day_of_week' column in your original dataset
if 'day_of_week' in combined_data_cleaned.columns:
    # Align 'day_of_week' with the test dataset indices
    results_df['DayOfWeek'] = combined_data_cleaned.loc[y_test.index, 'day_of_week']

    # Calculate the mean residuals grouped by the day of the week
    mean_error_by_day = results_df.groupby('DayOfWeek')['Residuals'].mean()

    # Plot the mean residuals by day of the week
    mean_error_by_day.plot(kind='bar', color='teal', alpha=0.7)
    plt.title("Mean Residuals by Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Mean Residual")
    plt.grid(axis='y')
    plt.show()
else:
    print("'day_of_week' column not found in combined_data_cleaned.")



# Plot distribution of residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.title('Error Distribution (Residuals) of Predictions')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


import xgboost as xgb

# Retrieve feature importance
xgb.plot_importance(random_search.best_estimator_, importance_type='weight', max_num_features=10, height=0.8)
plt.title("Feature Importance (XGBoost)")
plt.show()


combined_data_cleaned['datetime'] = pd.to_datetime(combined_data_cleaned['datetime'])  # Convert to datetime
combined_data_cleaned['weekday'] = combined_data_cleaned['datetime'].dt.weekday  # Extract weekday
combined_data_cleaned['hour'] = combined_data_cleaned['datetime'].dt.hour  # Extract hour


# Assuming 'combined_data_cleaned' has a column 'weekday' (0=Monday, 6=Sunday)
combined_data_residuals = pd.DataFrame({
    'weekday': combined_data_cleaned['weekday'].iloc[len(combined_data_cleaned) - len(residuals):],  # align residuals with 'weekday'
    'residuals': residuals
})

plt.figure(figsize=(12, 6))
plt.scatter(combined_data_residuals['weekday'], combined_data_residuals['residuals'], alpha=0.5)
plt.title('Residuals by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Residuals')
plt.show()


# Check if 'city_id' column exists in the combined data
if 'city_id' in combined_data_cleaned.columns:
    # Align 'city_id' column with the test dataset indices
    results_df['City_ID'] = combined_data_cleaned.loc[y_test.index, 'city_id']

    # Calculate the mean residuals grouped by city_id
    mean_error_by_city = results_df.groupby('City_ID')['Residuals'].mean()

    # Plot the mean residuals by city_id
    mean_error_by_city.plot(kind='bar', color='skyblue', alpha=0.8, figsize=(12, 6))
    plt.title("Mean Residuals by City_ID")
    plt.xlabel("City_ID")
    plt.ylabel("Mean Residual")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("'city_id' column not found in combined_data_cleaned.")


import matplotlib.pyplot as plt

# Ensure 'datetime' is converted to pandas datetime type
combined_data_cleaned['datetime'] = pd.to_datetime(combined_data_cleaned['datetime'])

# Align LSTM and hybrid predictions with the 'datetime' column
lstm_preds_full = np.pad(lstm_preds, (0, len(combined_data_cleaned) - len(lstm_preds)), constant_values=np.nan)
hybrid_preds_full = np.pad(lstm_preds_test, (0, len(combined_data_cleaned) - len(lstm_preds_test)), constant_values=np.nan)

# Plot predictions over time
plt.figure(figsize=(14, 7))
plt.plot(combined_data_cleaned['datetime'], combined_data_cleaned['target_7_days_ahead'], label='True PM2.5', color='blue', alpha=0.6)
plt.plot(combined_data_cleaned['datetime'], lstm_preds_full, label='LSTM Predictions', color='red', alpha=0.6)
plt.plot(combined_data_cleaned['datetime'], hybrid_preds_full, label='Hybrid Predictions', color='green', alpha=0.6)
plt.title('Prediction Over Time (True vs. Predicted)')
plt.xlabel('Time')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.show()


# Compute correlation matrix for numeric data
correlation_matrix = combined_data_cleaned.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Pollutants and Features")
plt.show()


import numpy as np

# Set a correlation threshold for filtering (optional, e.g., show only correlations > |0.3|)
correlation_threshold = 0.3
filtered_corr = correlation_matrix.copy()
filtered_corr[np.abs(filtered_corr) < correlation_threshold] = 0

# Plot heatmap with improvements
plt.figure(figsize=(16, 12))  # Increase figure size
sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, linecolor='gray')  # Add gridlines for clarity
plt.title("Correlation Heatmap of Pollutants and Features", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for readability
plt.yticks(fontsize=10)
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# Example: Trend of PM2.5 across time
plt.figure(figsize=(14, 6))
plt.plot(combined_data_cleaned['datetime'], combined_data_cleaned['PM2.5'], label='PM2.5')
plt.title("PM2.5 Levels Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.legend()
plt.grid(True)
plt.show()

# Define models and RMSEs
models = ['Linear Regression', 'Random Forest', 'XGBoost']
rmses = [rmse_lr, rmse_rf, rmse_xgb]

# Plot comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.bar(models, rmses, color='skyblue')
plt.title("Model Performance Comparison (RMSE)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.grid(axis='y')
plt.show()

import shap

# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_test)
