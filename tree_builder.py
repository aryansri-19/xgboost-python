import numpy as np
from typing import Tuple, Optional
from tree import TreeNode, SimpleTree

class SimpleTreeBuilder:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 2, reg_lambda: float = 1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda

    def find_best_split_exact(self, data: np.ndarray, indices: np.ndarray, grads: np.ndarray, hesses: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -float('inf')
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        node_grads = grads[indices]
        node_hesses = hesses[indices]

        parent_sum_grad = np.sum(node_grads)
        parent_sum_hess = np.sum(node_hesses)
        parent_score = (parent_sum_grad ** 2) / (parent_sum_hess + self.reg_lambda)

        n_features = data.shape[1]
        for feature_idx in range(n_features):
            feature_values = data[indices, feature_idx]

            sorted_indices = np.argsort(feature_values)
            sorted_grads = node_grads[sorted_indices]
            sorted_hesses = node_hesses[sorted_indices]
            sorted_features = feature_values[sorted_indices]

            left_sum_grad = 0.0
            left_sum_hess = 0.0

            for i in range(len(sorted_grads) - 1):
                left_sum_grad += sorted_grads[i]
                left_sum_hess += sorted_hesses[i]

                current_val = sorted_features[i]
                next_val = sorted_features[i+1]
                if current_val == next_val:
                    continue

                right_sum_grad = parent_sum_grad - left_sum_grad
                right_sum_hess = parent_sum_hess - left_sum_hess

                if left_sum_hess < 1e-3 or right_sum_hess < 1e-3:
                    continue

                left_score = (left_sum_grad ** 2) / (left_sum_hess + self.reg_lambda)
                right_score = (right_sum_grad ** 2) / (right_sum_hess + self.reg_lambda)
                gain = left_score + right_score - parent_score

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = (current_val + next_val) / 2

        return best_feature, best_threshold, best_gain

    def calculate_leaf_value(self, indices: np.ndarray, grads: np.ndarray, hesses: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0

        leaf_grads = grads[indices]
        leaf_hesses = hesses[indices]

        return -np.sum(leaf_grads) / (np.sum(leaf_hesses) + self.reg_lambda)

    def build_tree(self, data: np.ndarray, grads: np.ndarray, hesses: np.ndarray) -> SimpleTree:
        tree = SimpleTree()
        initial_indices = np.arange(data.shape[0])
        tree.root = self._build_recursive(data, initial_indices, grads, hesses, depth=0)
        return tree

    def _build_recursive(self, data: np.ndarray, indices: np.ndarray, grads: np.ndarray, hesses: np.ndarray, depth: int) -> TreeNode:
        if depth >= self.max_depth or len(indices) < self.min_samples_split:
            leaf_value = self.calculate_leaf_value(indices, grads, hesses)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        feature_idx, threshold, gain = self.find_best_split_exact(data, indices, grads, hesses)

        if feature_idx is None or gain <= 0:
            leaf_value = self.calculate_leaf_value(indices, grads, hesses)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        feature_values = data[indices, feature_idx]
        left_mask = feature_values < threshold

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        left_child = self._build_recursive(data, left_indices, grads, hesses, depth + 1)
        right_child = self._build_recursive(data, right_indices, grads, hesses, depth + 1)

        return TreeNode(
            feature_id=feature_idx,
            threshold=threshold,
            left_child=left_child,
            right_child=right_child,
            is_leaf=False
        )
