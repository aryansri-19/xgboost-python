import pandas as pd
import numpy as np
from utils import train_xgboost_on_dataframe, train_official_xgboost_on_dataframe, evaluate_model
import time

def main():
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
    })
    data['target'] = 2 * data['feature1'] + data['feature2'] + np.random.randn(1000) * 0.1

    train_data = data.iloc[:800]
    test_data = data.iloc[800:]
    
    feature_columns = ['feature1', 'feature2', 'feature3']

    print("--- Training Our Custom XGBoost Model ---")
    start_time = time.time()
    my_model = train_xgboost_on_dataframe(
        train_data,
        'target',
        feature_columns=feature_columns,
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        verbose=100
    )
    my_model_time = time.time() - start_time
    print(f"Custom model training time: {my_model_time:.2f}s\n")

    my_train_metrics = evaluate_model(my_model, train_data.loc[:, feature_columns], train_data.loc[:, 'target'])
    my_test_metrics = evaluate_model(my_model, test_data.loc[:, feature_columns], test_data.loc[:, 'target'])

    print("Custom Model Performance:")
    print(f"  Train RMSE: {my_train_metrics['rmse']:.4f}")
    print(f"  Test RMSE:  {my_test_metrics['rmse']:.4f}")
    print(f"  Train R-squared: {my_train_metrics['r2']:.4f}")
    print(f"  Test R-squared:  {my_test_metrics['r2']:.4f}\n")


    print("--- Training Official XGBoost Model ---")
    start_time = time.time()
    official_model = train_official_xgboost_on_dataframe(
        train_data,
        'target',
        feature_columns=feature_columns,
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        random_state=42
    )
    official_model_time = time.time() - start_time
    print(f"Official model training time: {official_model_time:.2f}s\n")

    official_train_metrics = evaluate_model(official_model, test_data.loc[:, feature_columns], test_data.loc[:, 'target'])
    official_test_metrics = evaluate_model(official_model, test_data.loc[:, feature_columns], test_data.loc[:, 'target'])

    print("Official Model Performance:")
    print(f"  Train RMSE: {official_train_metrics['rmse']:.4f}")
    print(f"  Test RMSE:  {official_test_metrics['rmse']:.4f}")
    print(f"  Train R-squared: {official_train_metrics['r2']:.4f}")
    print(f"  Test R-squared:  {official_test_metrics['r2']:.4f}\n")

if __name__ == "__main__":
    main()