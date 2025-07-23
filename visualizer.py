import graphviz
from typing import List
from tree import TreeNode, SimpleTree

def plot_tree(tree: SimpleTree, feature_names: List[str], filename: str = 'tree_visualization'):
    """
    Generates and saves a visualization of a single decision tree.

    Args:
        tree (SimpleTree): The tree to visualize.
        feature_names (List[str]): A list of the feature names used in the model.
        filename (str): The name of the output file (without extension).
    """
    if tree.root is None:
        print("Tree is empty, cannot visualize.")
        return

    # Create a new directed graph
    dot = graphviz.Digraph(comment='Decision Tree', format='png')
    dot.attr('node', shape='box', style='filled, rounded', color='lightblue2')
    dot.attr('edge', fontname='helvetica', fontsize='10')
    dot.attr(label=f'Decision Tree Visualization', fontsize='16')

    # A counter to give each node a unique ID
    node_counter = 0

    def add_nodes_edges(node: TreeNode, parent_id: str = 'root'):
        nonlocal node_counter
        my_id = str(node_counter)
        node_counter += 1

        if node.is_leaf:
            # Leaf node
            leaf_label = f"value = {node.leaf_value:.4f}"
            dot.node(my_id, label=leaf_label, shape='ellipse', style='filled', color='lightgreen')
        else:
            # Split node
            feature_name = feature_names[node.feature_id]
            threshold = node.threshold
            split_label = f"{feature_name} < {threshold:.3f}"
            dot.node(my_id, label=split_label)

            # Recurse for left child (True condition)
            if node.left_child:
                left_child_id = add_nodes_edges(node.left_child, my_id)
                dot.edge(my_id, left_child_id, label="True")

            # Recurse for right child (False condition)
            if node.right_child:
                right_child_id = add_nodes_edges(node.right_child, my_id)
                dot.edge(my_id, right_child_id, label="False")

        return my_id

    # Start the traversal from the root
    add_nodes_edges(tree.root)

    # Render the graph to a file and automatically open it
    try:
        dot.render(filename, view=True, cleanup=True)
        print(f"Tree visualization saved to '{filename}.png' and opened.")
    except Exception as e:
        print(f"Error rendering graph. Make sure Graphviz is installed and in your PATH.")
        print(f"Error details: {e}")
