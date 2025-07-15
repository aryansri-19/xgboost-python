import numpy as np
from typing import Tuple

class Objective:
    """Base class for objective functions"""
    def get_gradients(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class SquaredLossObjective(Objective):
    """Squared loss for regression"""
    def get_gradients(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates gradient and hessian for squared loss"""
        grads = 2 * (predictions - labels)
        hesses = np.full(len(labels), 2.0)
        return grads, hesses

class LogisticLossObjective(Objective):
    """Logistic loss for binary classification"""
    def get_gradients(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates gradient and hessian for logistic loss"""
        probs = 1 / (1 + np.exp(-predictions))
        grads = probs - labels
        hesses = probs * (1 - probs)
        return grads, hesses