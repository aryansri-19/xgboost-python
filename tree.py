import pandas as pd  
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
      
class SimpleTree:  
    """Basic decision tree for gradient boosting"""  
      
    def __init__(self):  
        self.root: Optional[TreeNode] = None  
          
    def predict_single(self, row: pd.Series) -> float:  
        """Predict single row through tree traversal"""  
        if self.root is None:  
            return 0.0  
              
        node: Optional[TreeNode] = self.root  
        while node and not node.is_leaf:  
            if node.feature_id is None or node.threshold is None:
                #can raise an error instead
                return 0.0

            feature_name = row.index[node.feature_id]  
            if pd.isna(row[feature_name]) or row[feature_name] < node.threshold:  
                node = node.left_child  
            else:  
                node = node.right_child  

        if node is None or node.leaf_value is None:
            return 0.0
            
        return node.leaf_value  
      
    def predict(self, data: pd.DataFrame) -> np.ndarray:  
        """Predict multiple rows"""  
        return np.array([self.predict_single(row) for _, row in data.iterrows()])