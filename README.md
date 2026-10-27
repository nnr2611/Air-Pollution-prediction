# Beijing Air Quality Prediction Project
## Overview 
This project aims to predict air quality indicators, such as PM2.5, in Beijing for the next 7 days using historical data and advanced machine learning techniques. The solution combines the temporal modeling capabilities of Long Short-Term Memory (LSTM) networks with the regression strength of XGBoost to provide robust and accurate predictions.

The project includes a well-structured pipeline for data preprocessing, feature engineering, model training, and evaluation. It is designed for scalability and reproducibility, adhering to high standards of code quality.

## **Repository Structure**

```plaintext
📂Air-Pollution-prediction
├── README.md
├── Model-Card.md
├── Architecture Diagram.jpeg
├── 📂data/
│   ├── raw/
├──📂 notebooks/
│   ├── ProjectML.ipynb
├── 📂src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── 📂results/
│   ├── figures/
├── LICENSE
└── requirements.txt
```

## How to use
### 1. Prerequisites:
Clone the repository and install the dependencies:
```
git clone https://github.com/yourusername/beijing-air-quality-predictor.git  
cd beijing-air-quality-predictor  
pip install -r requirements.txt  
```
### 2. Data Preparation
1. Download the [Beijing Multi-Site Air Quality Dataset](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set/code) from Kaggle. 
2. Place the raw dataset files in the ```data/raw/``` directory.

### 3. Running the Pipeline
```
python train.py --horizon 24     # forecast 24 hours ahead
python train.py --horizon 168    # forecast 7 days ahead
```
Outputs go to the directory given by `--out-dir` (`model.joblib`, `lstm.keras`, `metrics.json`, `test_predictions.csv`).
The dashboard expects one folder per horizon:
```
python train.py --horizon 24  --out-dir artifacts/h24
python train.py --horizon 168 --out-dir artifacts/h168
```

### 4. Dashboard
```
streamlit run dashboard/app.py
```
Pick the horizon, models, station and date range; tabs show forecast vs actual, model comparison against baselines, error analysis (by station, residuals) and feature importance, with CSV download.
The original exploratory LSTM/XGBoost hybrid lives in `notebook/ProjectML.ipynb`.

### Key Files
- `train.py`: end-to-end CLI (load, clean, features, train, evaluate, save).
- `src/preprocessing.py`: per-station cleaning on a regular hourly grid; chronological split.
- `src/feature_engineering.py`: lags, shifted rolling means, cyclical time and wind features; per-station target.
- `src/model_training.py`: XGBoost with early stopping; MAE/RMSE/R2/sMAPE.

## Current Results (chronological hold-out, last ~20% of the period, all 12 stations)
| Horizon | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| 24 h | XGBoost | 51.1 | 75.7 | 0.173 |
| 24 h | LSTM | 50.8 | 81.0 | 0.052 |
| 24 h | Persistence | 57.3 | 87.6 | -0.107 |
| 168 h | XGBoost | 58.2 | 82.4 | 0.025 |
| 168 h | LSTM | 55.9 | 88.0 | -0.111 |
| 168 h | Hybrid (10% LSTM) | 57.6 | 82.4 | 0.024 |

The LSTM has the lowest MAE but a higher RMSE: it predicts typical values well and underestimates
pollution spikes. The hybrid blend (weight tuned on a validation slice) is about equal to XGBoost.

Skill beyond a day is low: PM2.5 depends heavily on future weather, which the model does not see.
The earlier "20% RMSE reduction" claim came from the notebook, whose target was shifted across
interleaved stations, so it is not reproduced here.

## Next Steps
- Model Deployment: Package the system into an API for real-time predictions.
- Dashboard Integration: Build a web-based dashboard for visualization.
- Extended Features: Integrate additional external datasets (e.g., weather conditions).
