import pandas as pd
import argparse
from my_xgboost import SimpleXGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils import evaluate_classification_model
import time
from visualizer import plot_tree

def main():
    parser = argparse.ArgumentParser(description="XGBoost Example")
    parser.add_argument("--train_input", type=str, help="Path to the train CSV file", required=True)
    parser.add_argument("--valid_input", type=str, help="Path to the valid CSV file", required=True)
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_input, header=0)
    valid_data = pd.read_csv(args.valid_input, header=0)

    feature_columns = train_data.columns.tolist()[:-1]
    target_column = train_data.columns.tolist()[-1]

    print("--- Training Our Custom XGBoost Model ---")
    start_time = time.time()
    my_model = SimpleXGBoost(
        n_estimators=10,
        learning_rate=0.3,
        max_depth=4,
        verbose=1,
        objective='binary:logistic',
        n_bins=256
    )

    my_model.fit(train_data[feature_columns], train_data[target_column])
    my_model_time = time.time() - start_time
    print(f"Custom model training time: {my_model_time:.2f}s\n")

    my_train_metrics = evaluate_classification_model(my_model, train_data.loc[:, feature_columns], train_data.loc[:, target_column])
    my_test_metrics = evaluate_classification_model(my_model, valid_data.loc[:, feature_columns], valid_data.loc[:, target_column])

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
        n_estimators=10,
        learning_rate=0.3,
        max_depth=4,
        random_state=42,
        objective='binary:logistic',
        tree_method='hist'
    )
    official_model.fit(train_data[feature_columns], train_data[target_column])
    official_model_time = time.time() - start_time
    print(f"Official model training time: {official_model_time:.2f}s\n")

    official_train_metrics = evaluate_classification_model(official_model, train_data.loc[:, feature_columns], train_data.loc[:, target_column])
    official_test_metrics = evaluate_classification_model(official_model, valid_data.loc[:, feature_columns], valid_data.loc[:, target_column])

    print("Official Model Performance:")
    print(f"  Train Accuracy: {official_train_metrics['accuracy']:.4f}")
    print(f"  Test Accuracy:  {official_test_metrics['accuracy']:.4f}\n")

if __name__ == "__main__":
    main()
