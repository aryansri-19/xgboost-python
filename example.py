import pandas as pd
import numpy as np
from utils import train_xgboost_on_dataframe, train_official_xgboost_on_dataframe, evaluate_model
import time

def main():
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(10000),
        'feature2': np.random.randn(10000),
        'feature3': np.random.randn(10000),
        'feature4': np.random.randn(10000),
        'feature5': np.random.randn(10000),
        'feature6': np.random.randn(10000),
        'feature7': np.random.randn(10000),
        'feature8': np.random.randn(10000),
        'feature9': np.random.randn(10000),
        'feature10': np.random.randn(10000),
    })
    data['target'] = 4 * data['feature1'] + 2 * data['feature2'] + 1 * data['feature3'] + 0.5 * data['feature4'] + 0.25 * data['feature5'] + 0.125 * data['feature6'] + 0.0625 * data['feature7'] + 0.03125 * data['feature8'] + 0.015625 * data['feature9'] + 0.0078125 * data['feature10'] + np.random.randn(10000) * 0.1

    train_data = data.iloc[:8000]
    test_data = data.iloc[8000:]
    
    feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10']

    print("--- Training Our Custom XGBoost Model ---")
    start_time = time.time()
    my_model = train_xgboost_on_dataframe(
        train_data,
        'target',
        feature_columns=feature_columns,
        n_estimators=1000,
        learning_rate=0.3,
        max_depth=6,
        verbose=100,
        objective='reg:squarederror'
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
        learning_rate=0.3,
        max_depth=6,
        random_state=42,
        tree_method='exact'
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