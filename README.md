# Air-Pollution-prediction
Particulate matter pollution prediction using historical data from Beijing Weather Station 


## Hybrid LSTM-XGBoost Air Quality Prediction System

### Overview
This project implements a hybrid machine learning system combining Long Short-Term Memory (LSTM) networks and XGBoost models to predict air quality across multiple regions in Beijing. The system utilizes temporal patterns captured by LSTM and residual learning with XGBoost for robust forecasting.

### Key Features
- Temporal Data Handling: LSTM for sequential data modelling.
- Residual Learning: XGBoost fine-tunes predictions based on LSTM residuals.
- Feature Engineering: Spatial and temporal features enhance model performance.
- Extensive Evaluation: Performance analyzed across multiple metrics and temporal dimensions.


### **Repository Structure**

```plaintext
📂Air-Pollution-prediction
├── README.md
├── Model-Card.md
├── 📂data/
│   ├── raw/
│   ├── processed/
├──📂 notebooks/
│   ├── ProjectML.ipynb
│   ├── analysis_notebook.ipynb
├── 📂src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── 📂results/
│   ├── figures/
│   ├── metrics/
└── requirements.txt
```

### Getting Started
Follow these steps to set up and run the project.

#### 1. Prerequisites:
Install the required Python packages by running:
```
pip install -r requirements.txt
```
#### 2. Data Preparation:
1. Download the [Beijing Multi-Site Air Quality Dataset.](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set/code)
2. Place the raw data files in the ```data/raw/``` directory.
3. Preprocess the data:
```
python src/data_preprocessing.py
```
#### 3. Train the Model:
Run the training pipeline notebook:
```
notebooks/train_pipeline.ipynb
```
#### 4. Evaluate the Model
Use the evaluation pipeline notebook to assess model performance:
```
notebooks/evaluate_pipeline.ipynb
```

### Key Files
- ```data_preprocessing.py```: Handles data cleaning and feature engineering.
- ```model.py```: Implements the hybrid LSTM-XGBoost model.
- ```evaluation.py```: Includes functions to compute RMSE, MAPE, and trend detection accuracy.
- ```train_pipeline.ipynb```: Demonstrates training and validation workflows.
- ```model_card.md```: Provides detailed documentation on the final model.

### Results
The hybrid model achieves:
- High accuracy in forecasting air quality 7 days in advance.
- Robust performance across spatial (city/region) and temporal (day/week/year) dimensions.

### Next Steps
- Model Deployment: Package the system into an API for real-time predictions.
- Dashboard Integration: Build a web-based dashboard for visualization.
- Extended Features: Integrate additional external datasets (e.g., weather conditions).
