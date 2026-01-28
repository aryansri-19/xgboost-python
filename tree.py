import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class TreeNode:
    """Simple tree node structure"""
    feature_id: Optional[int] = None
    threshold: Optional[float] = None
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None
    leaf_value: Optional[float] = None
    is_leaf: bool = False
    default_left: bool = True

class SimpleTree:
    def __init__(self):
        self.root: Optional[TreeNode] = None

    def predict(self, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]
        predictions = np.zeros(n_samples)
        self._traverse_tree(self.root, data, np.arange(n_samples), predictions)

        return predictions

    def _traverse_tree(self, node: Optional[TreeNode], data: np.ndarray, indices: np.ndarray, predictions: np.ndarray):
        """Helper function to recursively traverse the tree with data indices."""
        if node is None or len(indices) == 0:
            return

        if node.is_leaf:
            predictions[indices] = node.leaf_value
            return

        if node.feature_id is None or node.threshold is None:
            return
        feature_data = data[indices, node.feature_id]

        nan_mask = np.isnan(feature_data)
        left_mask = feature_data < node.threshold

        if node.default_left:
            left_mask = left_mask | nan_mask

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        self._traverse_tree(node.left_child, data, left_indices, predictions)
        self._traverse_tree(node.right_child, data, right_indices, predictions)
