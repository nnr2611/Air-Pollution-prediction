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
Run the training and evaluation pipeline:
1. Open the ```notebooks/ProjectML.ipynb``` in your preferred environment (e.g., Jupyter, Colab).
2. Follow the steps in the notebook to train the models and evaluate performance.

### Key Files
- ```preprocessing.py```: Handles data cleaning and feature engineering.
- ```model_training.py```: Implements the hybrid LSTM-XGBoost model.
- ```ProjectML.ipynb```: Demonstrates training and validation workflows.
- ```model_card.md```: Provides detailed documentation on the final model.


## Features
- Preprocessing: Handles missing values, aggregates data to city and region levels, and normalizes features.
- Feature Engineering: Includes lagged features, rolling averages, and temporal indicators.
- Modeling: Combines LSTMs for temporal dependencies and XGBoost for static feature learning.
- Evaluation: Uses RMSE, MAPE, and trend accuracy for comprehensive evaluation.

## Key Results
- The ensemble model outperformed baselines (e.g., Linear Regression and ARIMA) by reducing RMSE by 20%.
- Captured trends in air quality changes effectively, aiding decision-making.

## Next Steps
- Model Deployment: Package the system into an API for real-time predictions.
- Dashboard Integration: Build a web-based dashboard for visualization.
- Extended Features: Integrate additional external datasets (e.g., weather conditions).
