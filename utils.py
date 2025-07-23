import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import accuracy_score
import xgboost as xgb
from my_xgboost import SimpleXGBoost
import time
EvaluableModel = Union[SimpleXGBoost, xgb.XGBRegressor]

def evaluate_regression_model(model: EvaluableModel, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance"""
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)
    mae = np.mean(np.abs(y - predictions))

    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'r2': r2
    }

def evaluate_classification_model(model: EvaluableModel, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance"""
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Predicting Time Duration: {duration:.2f} seconds")
    binary_predictions = (predictions > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y, binary_predictions)
    }
