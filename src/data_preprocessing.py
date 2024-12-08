import pandas as pd
import numpy as np

# Create a datetime column
combined_data['datetime'] = pd.to_datetime(combined_data[['year', 'month', 'day', 'hour']])

# Sort the dataset by datetime
combined_data = combined_data.sort_values(by='datetime')

# Verify the update
print("Data sorted by datetime successfully.")
print(combined_data[['datetime', 'year', 'month', 'day', 'hour']].head())

# Sort data by time for temporal consistency
combined_data = combined_data.sort_values(by='datetime')

# Inspect missing values
missing_summary = combined_data.isnull().sum()

# Drop columns with too many missing values or irrelevant ones
threshold = 0.5 * len(combined_data)  # Drop columns with >50% missing
combined_data_cleaned = combined_data.dropna(axis=1, thresh=threshold)

# Fill remaining missing values with forward fill or a constant
combined_data_cleaned = combined_data_cleaned.fillna(method='ffill').fillna(0)

# Feature Engineering
# Add temporal features
combined_data_cleaned['hour'] = combined_data_cleaned['datetime'].dt.hour
combined_data_cleaned['day'] = combined_data_cleaned['datetime'].dt.day
combined_data_cleaned['month'] = combined_data_cleaned['datetime'].dt.month
combined_data_cleaned['year'] = combined_data_cleaned['datetime'].dt.year
combined_data_cleaned['day_of_week'] = combined_data_cleaned['datetime'].dt.dayofweek
combined_data_cleaned['is_weekend'] = combined_data_cleaned['day_of_week'].isin([5, 6]).astype(int)

# Add a target variable for 7-day ahead prediction
combined_data_cleaned['target_7_days_ahead'] = combined_data_cleaned['PM2.5'].shift(-7)  # Shift PM2.5 by 7 days

# Add spatial granularity (e.g., site-level feature, which corresponds to city_id)
# Assuming 'site' column exists in the dataset
combined_data_cleaned['city_id'] = combined_data_cleaned['site'].astype('category').cat.codes

# Generate lag features for AQI or other pollutants
lag_features = ['PM2.5', 'PM10', 'O3']  # Replace with relevant column names
for feature in lag_features:
    for lag in range(1, 8):  # Create lag features for up to 7 days
        combined_data_cleaned[f'{feature}_lag_{lag}'] = combined_data_cleaned[feature].shift(lag)

# Generate rolling statistics
for feature in lag_features:
    combined_data_cleaned[f'{feature}_rolling_mean'] = combined_data_cleaned[feature].rolling(window=7).mean()
    combined_data_cleaned[f'{feature}_rolling_std'] = combined_data_cleaned[feature].rolling(window=7).std()

# Drop rows with NaN values introduced by rolling or lagging
combined_data_cleaned = combined_data_cleaned.dropna()

# Save cleaned and feature-engineered data to a CSV for further use
combined_data_cleaned.to_csv('processed_beijing_airquality_data.csv', index=False)

# Output dataset structure and head
print("Processed Dataset Info:")
print(combined_data_cleaned.info())
print(combined_data_cleaned.head())
