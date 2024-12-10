# Model Card: Beijing Air Quality Predictor
## Model Details
- Model Name: Beijing Air Quality Predictor
- Model Version: v1.0
- Model Type: Regression (Time-Series Prediction)
- Developers: Onkar Joshi , Nithish Ragav Narayana Shankar , Vanshika Parikh
- Release Date: 10-11-2024
## Intended Use
### Primary Use
The model is designed to predict air quality indicators (PM2.5) 7 days in advance at city and region levels using temporal and non-temporal features.

### Intended Users
- Environmental scientists and researchers studying air pollution trends.
- Policymakers and urban planners for decision-making.
- Public health organizations for planning interventions.
### Out-of-Scope Use Cases
- Predicting short-term variations (less than one day).
- Use for real-time air quality monitoring.
- Application in regions without similar air quality data or environmental factors.
## Model/Data Description
### Data Used
The model was trained on the Beijing Multi-Site Air Quality Dataset from Kaggle. The dataset includes hourly air quality readings from multiple monitoring sites in Beijing.

- Data Sources: Official government monitoring sites in Beijing.
- Preprocessing Steps:
  - Handling missing values via interpolation.
  - Aggregating multi-site data to region level.
  - Creating lagged and rolling features for time-series modeling.
- Biases:
  - Urban bias as rural areas may have different pollution sources.
  - Temporal bias due to seasonal or weather-related factors.
- Features
  - Temporal Features: Hour, day of the week, and month.
  - Lagged Features: Previous air quality measurements (lags of 1–7 days).
  - Rolling Features: Weekly averages for smoothing trends.
  - Static Features: Site-level and region-level identifiers.
## Model Architecture
- LSTM: Used to capture temporal dependencies.
- XGBoost: Used to model static features and perform feature-specific learning.
- Combined predictions from both models via ensembling.
## Training and Evaluation
### Training Procedure
- LSTMs were trained using TensorFlow/Keras with an Adam optimizer, batch size of 32, and a learning rate of 0.001.
- XGBoost was trained using 100 estimators, learning rate 0.1, and maximum tree depth of 5.
- Training was conducted on GPU-accelerated environments (e.g., Google Colab).
### Evaluation Metrics
- Root Mean Squared Error (RMSE).
- Mean Absolute Percentage Error (MAPE).
- Trend detection accuracy (based on comparison of predicted and actual trends).
### Baseline Comparison
- Simple linear regression and ARIMA models were used as baselines. The ensemble model outperformed these baselines by reducing RMSE by 20%.
## Ethical Considerations
### Fairness and Bias
- The dataset may reflect urban environments more than rural areas, limiting model applicability in less urbanized regions.
- Efforts were made to reduce bias by aggregating data across regions.
### Privacy
- The dataset does not include personally identifiable information (PII).
### Security
- Models and pipelines do not store sensitive data and are designed for transparency.
## Limitations and Recommendations
### Known Limitations
- Underperformance during anomalous weather events (e.g., sandstorms).
- Limited generalizability to regions outside of Beijing without retraining.
### Recommendations for Use
- Retrain the model on localized data for use in other regions.
- Avoid over-reliance on single-point predictions; use confidence intervals where applicable.
- Combine predictions with domain expertise for critical decision-making.
## Additional Information
- References:
  - Beijing Multi-Site Air Quality Data [Kaggle](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set/code)
### License:
None
## Contact Information:

- Onkar Joshi , Nithish Ragav Narayana Shankar , Vanshika Parikh
- Email: nithishr@ualberta.ca, oajoshi@ualberta.ca, parikh2@ualberta.ca
- GitHub:[link](https://github.com/nnr2611/Air-Pollution-prediction)
