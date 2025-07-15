import pandas as pd  
import numpy as np  
from typing import List  
from tree import SimpleTree  
from tree_builder import SimpleTreeBuilder  
from objectives import SquaredLossObjective, LogisticLossObjective  
import time
  
class SimpleXGBoost:  
    """Simplified XGBoost implementation"""  
      
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,   
                 max_depth: int = 6, objective: str = 'reg:squarederror', verbose: int = 10):  
        self.n_estimators = n_estimators  
        self.learning_rate = learning_rate  
        self.max_depth = max_depth  
        self.trees: List[SimpleTree] = []  
        self.base_score = 0.0  
        self.verbose = verbose
          
        # Set objective function  
        if objective == 'reg:squarederror':  
            self.objective = SquaredLossObjective()  
        elif objective == 'binary:logistic':  
            self.objective = LogisticLossObjective()  
        else:  
            raise ValueError(f"Unsupported objective: {objective}")  
          
        self.tree_builder = SimpleTreeBuilder(max_depth=max_depth)  
      
    def fit(self, X: pd.DataFrame, y: pd.Series):  
        """Train the XGBoost model"""  
        # Initialize base score  
        self.base_score = y.mean()  
        predictions = np.full(len(y), self.base_score)  
          
        # Boosting iterations  
        start_time = time.time()
        for iteration in range(self.n_estimators):  
            # Compute gradients  
            grads, hesses = self.objective.get_gradients(predictions, y.to_numpy())  
              
            # Build new tree  
            tree = SimpleTree()  
            tree.root = self.tree_builder.build_tree(X, grads, hesses)  
              
            # Update predictions  
            tree_predictions = tree.predict(X)  
            predictions += self.learning_rate * tree_predictions  
              
            self.trees.append(tree)  
            
            if self.verbose > 0 and (iteration + 1) % self.verbose == 0:
                print(f"[{iteration + 1}/{self.n_estimators}] epoch... Took {time.time() - start_time:.2f} seconds")
                start_time = time.time()
      
    def predict(self, X: pd.DataFrame) -> np.ndarray:  
        """Make predictions using ensemble of trees"""  
        predictions = np.full(len(X), self.base_score)  
          
        for tree in self.trees:  
            tree_predictions = tree.predict(X)  
            predictions += self.learning_rate * tree_predictions  
          
        return predictions