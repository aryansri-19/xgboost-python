import numpy as np
from typing import Tuple, Optional, List
from tree import TreeNode, SimpleTree

class SimpleTreeBuilder:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 2, reg_lambda: float = 1.0, n_bins: int = 256):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.n_bins = n_bins
        self.thresholds: List[np.ndarray] = []

    def set_thresholds(self, thresholds: List[np.ndarray]):
        self.thresholds = thresholds

    def find_best_split_hist(self, X_binned: np.ndarray, indices: np.ndarray, grads: np.ndarray, hesses: np.ndarray) -> Tuple[Optional[int], Optional[float], float, bool]:
        node_grads = grads[indices]
        node_hesses = hesses[indices]
        
        parent_sum_grad = np.sum(node_grads)
        parent_sum_hess = np.sum(node_hesses)
        parent_score = (parent_sum_grad ** 2) / (parent_sum_hess + self.reg_lambda)

        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X_binned.shape[1]

        for feature_idx in range(n_features):
            bin_ids = X_binned[indices, feature_idx]
            
            current_n_bins = len(self.thresholds[feature_idx]) + 1
            if current_n_bins <= 1:
                continue

            g_hist = np.bincount(bin_ids, weights=node_grads, minlength=current_n_bins)
            h_hist = np.bincount(bin_ids, weights=node_hesses, minlength=current_n_bins)

            gl_cumulative = np.cumsum(g_hist)
            hl_cumulative = np.cumsum(h_hist)

            gl = gl_cumulative[:-1]
            hl = hl_cumulative[:-1]
            
            gr = parent_sum_grad - gl
            hr = parent_sum_hess - hl

            valid_mask = (hl > 1e-3) & (hr > 1e-3)
            if not np.any(valid_mask):
                continue
            left_scores = (gl ** 2) / (hl + self.reg_lambda)
            right_scores = (gr ** 2) / (hr + self.reg_lambda)
            
            gains = left_scores + right_scores - parent_score
            gains[~valid_mask] = -float('inf')

            local_best_bin_idx = np.argmax(gains)
            local_max_gain = gains[local_best_bin_idx]

            if local_max_gain > best_gain:
                best_gain = local_max_gain
                best_feature = feature_idx
                if local_best_bin_idx < len(self.thresholds[feature_idx]):
                     best_threshold = self.thresholds[feature_idx][local_best_bin_idx]
                else:
                    best_threshold = self.thresholds[feature_idx][-1]

        return best_feature, best_threshold, best_gain, False

    def calculate_leaf_value(self, indices: np.ndarray, grads: np.ndarray, hesses: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        return -np.sum(grads[indices]) / (np.sum(hesses[indices]) + self.reg_lambda)

    def build_tree(self, X_binned: np.ndarray, grads: np.ndarray, hesses: np.ndarray) -> SimpleTree:
        tree = SimpleTree()
        initial_indices = np.arange(X_binned.shape[0])
        tree.root = self._build_recursive(X_binned, initial_indices, grads, hesses, depth=0)
        return tree

    def _build_recursive(self, X_binned: np.ndarray, indices: np.ndarray, grads: np.ndarray, hesses: np.ndarray, depth: int) -> TreeNode:
        if depth >= self.max_depth or len(indices) < self.min_samples_split:
            leaf_value = self.calculate_leaf_value(indices, grads, hesses)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        feature_idx, threshold, gain, default_left = self.find_best_split_hist(X_binned, indices, grads, hesses)

        if feature_idx is None or gain <= 0:
            leaf_value = self.calculate_leaf_value(indices, grads, hesses)
            return TreeNode(is_leaf=True, leaf_value=leaf_value)

        cut_float = threshold
        
        feat_thresholds = self.thresholds[feature_idx]
        cut_bin_idx = np.searchsorted(feat_thresholds, cut_float)
        
        bin_values = X_binned[indices, feature_idx]
        left_mask = bin_values <= cut_bin_idx
        
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        left_child = self._build_recursive(X_binned, left_indices, grads, hesses, depth + 1)
        right_child = self._build_recursive(X_binned, right_indices, grads, hesses, depth + 1)

        return TreeNode(
            feature_id=feature_idx,
            threshold=threshold,
            left_child=left_child,
            right_child=right_child,
            is_leaf=False,
            default_left=default_left
        )