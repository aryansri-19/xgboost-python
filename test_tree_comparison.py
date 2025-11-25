

import pandas as pd
import numpy as np
from my_xgboost import SimpleXGBoost
from sklearn.model_selection import train_test_split
from utils import evaluate_classification_model

def generate_random_data(n=10000):
    data = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    data['target'] = np.random.choice([0, 1], size=n)
    return data

def generate_learnable_data(n=10000):
    data = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    decision_boundary = data['feature1'] + 2 * data['feature2'] + 0.5 * data['feature3']
    data['target'] = (decision_boundary > 0).astype(int)
    
    noise_mask = np.random.rand(n) < 0.05
    data.loc[noise_mask, 'target'] = 1 - data.loc[noise_mask, 'target']
    return data

def test_config(train_data, test_data, n_estimators, max_depth, name):
    feature_columns = ['feature1', 'feature2', 'feature3']
    
    model = SimpleXGBoost(
        n_estimators=n_estimators,
        learning_rate=0.3,
        max_depth=max_depth,
        verbose=0,
        objective='binary:logistic'
    )
    
    model.fit(train_data[feature_columns], train_data['target'], objective='binary:logistic')
    train_metrics = evaluate_classification_model(model, train_data[feature_columns], train_data['target'])
    test_metrics = evaluate_classification_model(model, test_data[feature_columns], test_data['target'])
    
    print(f"{name:35s} → Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"{name:35s} → Test Accuracy: {test_metrics['accuracy']:.4f}")
    return test_metrics['accuracy']

def main():
    
    learnable_data = generate_learnable_data(n=1000000)
    train_learn, test_learn = train_test_split(learnable_data, test_size=0.2, random_state=42)
    
    test_config(train_learn, test_learn, 1, 11, "1 tree, depth=11")
    test_config(train_learn, test_learn, 136, 4, "136 tree, depth=4")
    test_config(train_learn, test_learn, 66, 5, "66 trees, depth=5")
    test_config(train_learn, test_learn, 32, 6, "32 trees, depth=6")

if __name__ == "__main__":
    main()
