import numpy as np
import pandas as pd
from typing import List, Optional, Union
import xgboost as xgb
from my_xgboost import SimpleXGBoost
EvaluableModel = Union[SimpleXGBoost, xgb.XGBRegressor]


def train_xgboost_on_dataframe(df: pd.DataFrame, target_column: str,
                               feature_columns: Optional[List[str]] = None, **kwargs) -> SimpleXGBoost:
    """Train our custom XGBoost on any pandas DataFrame"""

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]

    model = SimpleXGBoost(**kwargs)
    model.fit(X, y, objective=kwargs['objective'])

    return model

def train_official_xgboost_on_dataframe(df: pd.DataFrame, target_column: str,
                                         feature_columns: Optional[List[str]] = None, **kwargs) -> xgb.XGBRegressor:
    """Train official XGBoost on any pandas DataFrame"""

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]

    model = xgb.XGBRegressor(**kwargs)
    model.fit(X, y)

    return model


def evaluate_model(model: EvaluableModel, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model performance"""
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)
    mae = np.mean(np.abs(y - predictions))

    # Calculate R-squared
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