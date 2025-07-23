import pandas as pd
import numpy as np
from my_xgboost import SimpleXGBoost
from utils import evaluate_classification_model
import xgboost as xgb
import time
from visualizer import plot_tree

def main():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
    })
    data['target'] = np.random.choice([0, 1], size=1000)

    train_data = data.iloc[:800]
    test_data = data.iloc[800:]

    feature_columns = ['feature1', 'feature2', 'feature3']

    print("--- Training Our Custom XGBoost Model ---")
    start_time = time.time()
    my_model = SimpleXGBoost(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        verbose=100,
        objective='binary:logistic'
    )

    my_model.fit(train_data[feature_columns], train_data['target'], objective='binary:logistic')
    my_model_time = time.time() - start_time
    print(f"Custom model training time: {my_model_time:.2f}s\n")

    my_train_metrics = evaluate_classification_model   (my_model, train_data.loc[:, feature_columns], train_data.loc[:, 'target'])
    my_test_metrics = evaluate_classification_model(my_model, test_data.loc[:, feature_columns], test_data.loc[:, 'target'])

    print("Custom Model Performance:")
    print(f"  Train Accuracy: {my_train_metrics['accuracy']:.4f}")
    print(f"  Test Accuracy:  {my_test_metrics['accuracy']:.4f}\n")

    if my_model.trees:
        print("Visualizing the first decision tree...")
        first_tree = my_model.trees[0]
        # Plot it
        plot_tree(
            first_tree,
            feature_names=feature_columns,
            filename='my_first_xgboost_tree'
        )
    else:
        print("No trees were trained, skipping visualization.")

    print("--- Training Official XGBoost Model ---")
    start_time = time.time()
    official_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.3,
        max_depth=3,
        random_state=42,
        objective='binary:logistic'
    )
    official_model.fit(train_data[feature_columns], train_data['target'])
    official_model_time = time.time() - start_time
    print(f"Official model training time: {official_model_time:.2f}s\n")

    official_train_metrics = evaluate_classification_model(official_model, test_data.loc[:, feature_columns], test_data.loc[:, 'target'])
    official_test_metrics = evaluate_classification_model(official_model, test_data.loc[:, feature_columns], test_data.loc[:, 'target'])

    print("Official Model Performance:")
    print(f"  Train Accuracy: {official_train_metrics['accuracy']:.4f}")
    print(f"  Test Accuracy:  {official_test_metrics['accuracy']:.4f}\n")

if __name__ == "__main__":
    main()
