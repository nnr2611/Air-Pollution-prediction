# Model Card: Beijing Air Quality Predictor
## Model Details
- Model Name: Beijing Air Quality Predictor
- Model Version: v2.0
- Model Type: Regression (Time-Series Prediction)
- Developers: Onkar Joshi , Nithish Ragav Narayana Shankar , Vanshika Parikh
- Release Date: 10-11-2024 (revised 2026-10-27: pipeline rewritten, results re-measured)
## Intended Use
### Primary Use
Forecast hourly PM2.5 (µg/m³) at each of the 12 Beijing monitoring stations, 24 hours or 7 days (168 hours) ahead, for research and exploration.

### Intended Users
- Environmental scientists and researchers studying air pollution trends.
- Students and practitioners exploring air-quality forecasting baselines.

### Out-of-Scope Use Cases
- Operational health advice, alerts or policy decisions: skill is low (see results) and the data is 2013-2017.
- Real-time monitoring or horizons shorter than a day.
- Cities other than Beijing without retraining.
## Model/Data Description
### Data Used
Beijing Multi-Site Air Quality Dataset (Kaggle / UCI PRSA): hourly pollutant (PM2.5, PM10, SO2, NO2, CO, O3) and weather (temperature, pressure, dew point, rain, wind speed/direction) readings for 12 stations, 2013-03-01 to 2017-02-28.

- Preprocessing (per station, `src/preprocessing.py`):
  - Duplicates removed and the series placed on a regular hourly grid.
  - Gaps up to 6 hours are interpolated; longer gaps stay missing (XGBoost handles NaN; LSTM windows impute 0 after standardisation).
  - Wind direction converted to degrees and sin/cos.
- Features (`src/feature_engineering.py`):
  - Lags of all pollutants (1, 2, 3, 6, 12, 24, 48, 168 hours).
  - Trailing 24 h and 168 h rolling means (shifted so a row never sees its own value).
  - Cyclical hour/month encodings, day of week, weekend flag, station identifier.
- Target: PM2.5 at t + horizon, built per station so it never crosses stations.
- Biases:
  - Urban, Beijing-only coverage.
  - Strong seasonality (winter heating) with only ~4 years of data.
## Model Architecture
- **XGBoost** (`n_estimators` up to 600 with early stopping, learning rate 0.05, max depth 6) on the tabular features.
- **LSTM** (Keras): previous 48 hours of 17 channels plus a station embedding; 1 LSTM layer (16 units), dropout 0.5, L2, Gaussian input noise; trained on a standardised log1p target with Huber loss.
- **Hybrid:** weighted average of the two, with the weight chosen on a validation slice.
## Training and Evaluation
### Procedure
- Chronological split: the last 20% of time (2016-05 to 2017-02) is the hold-out test; the 15% of the remaining period just before it is used for early stopping and for choosing the blend weight. No shuffling.
- Reproduce with `python train.py --horizon 24` and `python train.py --horizon 168`.
### Metrics
MAE, RMSE, R² and sMAPE on the hold-out, pooled over all 12 stations.
### Results
| Horizon | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| 24 h | XGBoost | 51.1 | 75.7 | 0.173 |
| 24 h | LSTM | 50.8 | 81.0 | 0.052 |
| 24 h | Persistence (baseline) | 57.3 | 87.6 | -0.107 |
| 168 h | XGBoost | 58.2 | 82.4 | 0.025 |
| 168 h | LSTM | 55.9 | 88.0 | -0.111 |
| 168 h | Hybrid (10% LSTM) | 57.6 | 82.4 | 0.024 |
| 168 h | Persistence (baseline) | 75.5 | 112.5 | -0.816 |

(The 24 h hybrid is in `artifacts/h24/metrics.json`.)

- Both models beat naive persistence, but absolute skill is low, especially at 7 days (R² close to 0).
- The LSTM has the lowest MAE but a higher RMSE because it underestimates pollution spikes. XGBoost is better at high-pollution episodes.
- The hybrid is roughly equal to XGBoost.
- Earlier versions of this card and the README claimed a 20% RMSE reduction over linear regression and ARIMA. That figure came from the exploratory notebook, whose target was shifted across interleaved stations, and it is **not reproduced** here. ARIMA and linear regression baselines were not rerun.
## Ethical Considerations
### Fairness and Bias
1. Location bias: Beijing only; retrain before using elsewhere.
2. Temporal bias: data ends in February 2017 and does not reflect later emission policies or conditions.
### Privacy and Security
The dataset has no personal data. The model is not a safety-critical system and should not be used as one.
## Limitations and Recommendations
### Known Limitations
- No future weather input: PM2.5 depends strongly on forecast wind and humidity, which are unknown to the model. This is the main reason skill is low beyond a day.
- Underestimates extreme pollution events.
- Test metrics are pooled and cover mostly spring to winter; seasonal performance was not broken out.
### Recommendations
- Treat forecasts as rough guidance; do not use them as sole input for health decisions.
- Add weather-forecast features and retrain on recent data before any operational use.
## Additional Information
- References: [Beijing Multi-Site Air Quality Data](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set/code)
### License:
MIT License
## Contact Information:

- Onkar Joshi , Nithish Ragav Narayana Shankar , Vanshika Parikh
- Email: nithishr@ualberta.ca, oajoshi@ualberta.ca, parikh2@ualberta.ca
- GitHub:[link](https://github.com/nnr2611/Air-Pollution-prediction)
