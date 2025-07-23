import pandas as pd
import numpy as np
from typing import List
from tree import SimpleTree
from tree_builder import SimpleTreeBuilder
from objectives import SquaredLossObjective, LogisticLossObjective
import time

class SimpleXGBoost:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, objective: str = 'reg:squarederror', verbose: int = 10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: List[SimpleTree] = []
        self.base_score = 0.0
        self.verbose = verbose
        self.feature_names = []

        if objective == 'reg:squarederror':
            self.objective = SquaredLossObjective()
        else:
            self.objective = LogisticLossObjective()
        self.tree_builder = SimpleTreeBuilder(max_depth=max_depth)

    def fit(self, X: pd.DataFrame, y: pd.Series, objective: str):
        """Train the XGBoost model"""
        self.feature_names = X.columns.tolist()
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        if objective == 'reg:squarederror':
            self.base_score = y_np.mean()
        else:
            score = np.sum(y_np) / len(y_np)
            self.base_score = np.log(score / (1 - score))

        predictions = np.full(X_np.shape[0], self.base_score)

        start_time = time.time()
        for iteration in range(self.n_estimators):
            grads, hesses = self.objective.get_gradients(predictions, y_np)

            tree = self.tree_builder.build_tree(X_np, grads, hesses)

            tree_predictions = tree.predict(X_np)
            predictions += self.learning_rate * tree_predictions

            self.trees.append(tree)

            if self.verbose > 0 and (iteration + 1) % self.verbose == 0:
                print(f"[{iteration + 1}/{self.n_estimators}] epoch... Took {time.time() - start_time:.2f} seconds")
                start_time = time.time()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.to_numpy()
        predictions = np.full(X_np.shape[0], self.base_score)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X_np)

        return predictions
