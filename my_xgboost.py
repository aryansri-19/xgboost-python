import pandas as pd
import numpy as np
from typing import List, Tuple
from tree import SimpleTree
from tree_builder import SimpleTreeBuilder
from objectives import SquaredLossObjective, LogisticLossObjective
import time

class SimpleXGBoost:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, objective: str = 'reg:squarederror', 
                 verbose: int = 100, n_bins: int = 256):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.trees: List[SimpleTree] = []
        self.base_score = 0.0
        self.verbose = verbose
        self.feature_names = []
        self.bin_thresholds = []

        if objective == 'reg:squarederror':
            self.objective = SquaredLossObjective()
        else:
            self.objective = LogisticLossObjective()
        
        self.tree_builder = SimpleTreeBuilder(max_depth=max_depth, n_bins=n_bins)

    def _create_bins(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Maps continuous features to integer bins (0 to n_bins-1).
        Returns:
            X_binned: Integer array of bin indices.
            thresholds: List of arrays containing the float values of bin edges.
        """
        n_features = X.shape[1]
        thresholds = []
        X_binned = np.zeros_like(X, dtype=np.uint16)
        for i in range(n_features):
            col_data = X[:, i]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) == 0:
                unique_thresholds = np.array([])
            else:
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                unique_thresholds = np.percentile(valid_data, percentiles)
                unique_thresholds = np.unique(unique_thresholds)
                if len(unique_thresholds) > 2:
                    unique_thresholds = unique_thresholds[1:-1]
            
            thresholds.append(unique_thresholds)

            X_binned[:, i] = np.digitize(col_data, unique_thresholds)
        
        return X_binned, thresholds

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        print("Binning data for histogram optimization...")
        start_bin = time.time()
        X_binned, self.bin_thresholds = self._create_bins(X_np)
        print(f"Binning done in {time.time() - start_bin:.2f}s")

        self.tree_builder.set_thresholds(self.bin_thresholds)

        if self.objective == SquaredLossObjective():
            self.base_score = y_np.mean()
        else:
            score = np.sum(y_np) / len(y_np)
            score = np.clip(score, 1e-7, 1 - 1e-7)
            self.base_score = np.log(score / (1 - score))

        predictions = np.full(X_np.shape[0], self.base_score)

        start_time = time.time()
        for iteration in range(self.n_estimators):
            grads, hesses = self.objective.get_gradients(predictions, y_np)

            tree = self.tree_builder.build_tree(X_binned, grads, hesses)

            tree_predictions = tree.predict(X_np)
            predictions += self.learning_rate * tree_predictions

            self.trees.append(tree)

            if self.verbose > 0 and (iteration + 1) % self.verbose == 0:
                if self.objective == SquaredLossObjective():
                    train_loss = np.mean((predictions - y_np) ** 2)
                    print(f"[{iteration + 1}]\ttrain-rmse: {np.sqrt(train_loss):.6f}\ttime: {time.time() - start_time:.2f}s")
                else:
                    probs = 1 / (1 + np.exp(-predictions))
                    probs = np.clip(probs, 1e-7, 1 - 1e-7)
                    train_loss = -np.mean(y_np * np.log(probs) + (1 - y_np) * np.log(1 - probs))
                    accuracy = np.mean((probs >= 0.5) == y_np)
                    print(f"[{iteration + 1}]\ttrain-logloss: {train_loss:.6f}\ttrain-accuracy: {accuracy:.6f}\ttime: {time.time() - start_time:.2f}s")
                start_time = time.time()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.to_numpy()
        predictions = np.full(X_np.shape[0], self.base_score)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X_np)

        if self.objective == LogisticLossObjective():
            predictions = 1 / (1 + np.exp(-predictions))

        return predictions