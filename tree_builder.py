import pandas as pd
import numpy as np
from typing import Tuple, Optional
from tree import TreeNode

class SimpleTreeBuilder:
    """Basic tree construction algorithm"""

    def __init__(self, max_depth: int = 6, min_samples_split: int = 2, reg_lambda: float = 1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda

    def find_best_split(self, data: pd.DataFrame, grads: np.ndarray, hesses: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best feature and threshold for splitting"""
        best_gain = -float('inf')
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        
        parent_sum_grad = np.sum(grads)
        parent_sum_hess = np.sum(hesses)
        parent_score = (parent_sum_grad ** 2) / (parent_sum_hess + self.reg_lambda)

        for feature_idx, feature_name in enumerate(data.columns):
            feature_values = data[feature_name].to_numpy()
            
            sorted_indices = np.argsort(feature_values)
            sorted_grads = grads[sorted_indices]
            sorted_hesses = hesses[sorted_indices]
            sorted_features = feature_values[sorted_indices]

            left_sum_grad = 0.0
            left_sum_hess = 0.0
            
            for i in range(len(sorted_grads) - 1):
                left_sum_grad += sorted_grads[i]
                left_sum_hess += sorted_hesses[i]

                current_val = sorted_features[i]
                next_val = sorted_features[i+1]
                if pd.isna(current_val) or current_val == next_val:
                    continue

                if (i + 1) < self.min_samples_split or (len(sorted_grads) - (i + 1)) < self.min_samples_split:
                    continue
                
                right_sum_grad = parent_sum_grad - left_sum_grad
                right_sum_hess = parent_sum_hess - left_sum_hess

                left_score = (left_sum_grad ** 2) / (left_sum_hess + self.reg_lambda)
                right_score = (right_sum_grad ** 2) / (right_sum_hess + self.reg_lambda)
                
                gain = left_score + right_score - parent_score

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = (current_val + next_val) / 2

        return best_feature, best_threshold, best_gain

    def calculate_leaf_value(self, grads: np.ndarray, hesses: np.ndarray) -> float:
        """Calculate optimal leaf value"""
        if len(grads) == 0:
            return 0.0
        sum_grad = np.sum(grads)
        sum_hess = np.sum(hesses)
        return -sum_grad / (sum_hess + self.reg_lambda)

    def build_tree(self, data: pd.DataFrame, grads: np.ndarray, hesses: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build decision tree"""
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            leaf_value = self.calculate_leaf_value(grads, hesses)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        feature_idx, threshold, gain = self.find_best_split(data, grads, hesses)

        if feature_idx is None or gain <= 0:
            leaf_value = self.calculate_leaf_value(grads, hesses)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        feature_name = data.columns[feature_idx]
        left_mask = data[feature_name] < threshold
        right_mask = ~left_mask

        left_child = self.build_tree(data.loc[left_mask], grads[left_mask.values], hesses[left_mask.values], depth + 1)
        right_child = self.build_tree(data.loc[right_mask], grads[right_mask.values], hesses[right_mask.values], depth + 1)

        return TreeNode(
            feature_id=feature_idx,
            threshold=threshold,
            left_child=left_child,
            right_child=right_child,
            is_leaf=False
        )