from __future__ import annotations

# Standard Import
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import dendropy
import numpy as np

# Custom Import
from src.io.file import smart_open

# =================================================================
# Basic functionality
# =================================================================
def load(input_file: Path, schema: str = "newick") -> dendropy.Tree:
    """
    Loads a tree file with dendropy.

    Parameters
    ----------
    input_file     : file to load.
    schema         : format [.nwk .json .nexus .phyloxml].
    """
    with smart_open(input_file, "rt") as f:
        return dendropy.Tree.get(file=f, schema=schema, preserve_underscores=True)

def index_nodes(tree: dendropy.Tree, order: str = "preorder") -> list[dendropy.Node]:
    """
    Add numerical index to nodes (by transversal).

    Parameters
    ----------
    tree            : load dendropy object.
    order           : [preorder, inorder, postorder].
    """
    # Transversal
    match order:
        case "preorder":
            return list(tree.preorder_node_iter())
        case "inorder":
            return list(tree.inorder_node_iter())
        case "postorder":
            return list(tree.postorder_node_iter())
        case _:
            raise ValueError(f"Unknown traversal order: {order}!")

def index_root(pointer_parent_node: np.ndarray) -> int:
    """
    Retrieve the root index (the only -1)

    Parameters
    ----------
    pointer_parent_node : from create_pointer().
    """
    return int(np.flatnonzero(pointer_parent_node == -1)[0])

def extract_leaves(nodes: list[dendropy.Node]) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates leaf_label <-> node_index mapping.

    Parameters
    ----------
    nodes : nodes after index_nodes().

    Returns
    ----------
    1) leaf_labels
    2) leaf_index
    """
    leaves = [(i, node) for i, node in enumerate(nodes) if node.is_leaf()]
    leaf_labels  = np.array([node.taxon.label for _, node in leaves], dtype=object)
    leaf_ids     = np.array([i for i, _ in leaves], dtype=np.int64)
    return leaf_labels, leaf_ids

# =================================================================
# Pointers
# =================================================================
def create_pointer_parent(tree: dendropy.Tree, order: str = "preorder") -> tuple[list[dendropy.Node], np.ndarray, np.ndarray]:
    """
    Create condense array pointer of a tree.

    Parameters
    ----------
    tree            : load dendropy object.
    order           : [preorder, inorder, postorder].
    
    Returns
    ----------
    1) nodes                : dendropy.Node with additional "index" value.
    2) parent node pointer  : index = node, value = parent node.
    3) parent edge weight   : index = node, value = edge weight to parent.
    """
    nodes = index_nodes(tree, order)
    position = {node: i for i, node in enumerate(nodes)}
    
    # Pointer to parent node and edge weight
    parent_node = np.full(len(nodes), -1, dtype=np.int64)
    parent_edge = np.full(len(nodes), -1.0, dtype=np.float64)
    for i, node in enumerate(nodes):
        # Set edge weight
        parent_edge[i] = node.edge.length or 0.0
        
        # Set node pointer
        if node.parent_node is not None:
            parent_node[i] = position[node.parent_node]       
    return nodes, parent_node, parent_edge

# =================================================================
# Lift table & LCA
# =================================================================
def lift_table(parent: np.ndarray) -> np.ndarray:
    """Binary-lifting table for 2^n ancestor tracing."""
    n_nodes  = len(parent)
    n_levels = max(1, int(np.ceil(np.log2(n_nodes))))
    lift = np.full((n_nodes, n_levels), -1, dtype=np.int64)
    lift[:, 0] = parent
    for level in range(1, n_levels):
        previous = lift[:, level - 1]
        lift[:, level] = np.where(previous == -1, -1, lift[previous, level - 1])
    return lift

def lca(a: int, b: int, lift_table: np.ndarray, topological_depth: np.ndarray) -> int:
    """Lowest common ancestor of nodes a and b, via binary lifting."""
    if topological_depth[a] < topological_depth[b]:
        a, b = b, a
    diff = int(topological_depth[a] - topological_depth[b])
    for level in range(lift_table.shape[1]):
        if diff & (1 << level):
            a = lift_table[a, level]
    if a == b:
        return a
    for level in range(lift_table.shape[1] - 1, -1, -1):
        if lift_table[a, level] != lift_table[b, level]:
            a = lift_table[a, level]
            b = lift_table[b, level]
    return lift_table[a, 0] 

# =================================================================
# Distance
# =================================================================
def compute_relative_distance(source: int, pointer_parent_node: np.ndarray, pointer_parent_edge: np.ndarray) -> np.ndarray:
    """
    Compute relative distance from source to other nodes.

    Parameters
    ----------
    source                  : index of the target node.
    pointer_parent_node     : pointer to parent index.
    pointer_parent_edge     : edge length to parent from index.
    """    
    # Step 1: parent -> child mapping
    n_nodes = len(pointer_parent_node)
    pointer_child_node = [[] for _ in range(n_nodes)]
    for n in range(n_nodes):
        if pointer_parent_node[n] != -1:
            pointer_child_node[pointer_parent_node[n]].append(n)

    # Step 2: BFS out from source (down to children, up to parent)
    distance = np.full(n_nodes, -1.0, dtype=np.float64)
    distance[source] = 0.0
    q = deque([source])
    while q:
        n = q.popleft()
        # down: n -> child c, edge stored under child c
        for c in pointer_child_node[n]:
            if distance[c] < 0:
                distance[c] = pointer_parent_edge[c] + distance[n]
                q.append(c)
        # up: n -> parent p, edge stored under child n
        p = pointer_parent_node[n]
        if p != -1 and distance[p] < 0:
            distance[p] = pointer_parent_edge[n] + distance[n]
            q.append(p)
    return distance           
                
def compute_topological_depth(pointer_parent_node: np.ndarray) -> np.ndarray:
    """Number of edges from the root to each node."""
    return compute_relative_distance(index_root(pointer_parent_node), pointer_parent_node, np.ones(len(pointer_parent_node), dtype=np.float64))

def compute_rooted_distance(pointer_parent_node: np.ndarray, edge_length: np.ndarray) -> np.ndarray:
    """Cumulative branch length from the root to each node."""
    return compute_relative_distance(index_root(pointer_parent_node), pointer_parent_node, edge_length)

def patristic_distance(a: int, b: int, lift_table: np.ndarray, topological_depth: np.ndarray, root_distance: np.ndarray) -> float:
    """Sum of branch lengths on the tree path between nodes a and b."""
    ancestor = lca(a, b, lift_table, topological_depth)
    return float(root_distance[a] + root_distance[b] - 2 * root_distance[ancestor])

# =================================================================
# Rerooting Techniques
# =================================================================
def diameter_midpoint_distance(parent: np.ndarray, edge_length: np.ndarray, leaf_node_index: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Distance from each leaf to the midpoint of the tree's diameter.
    
    Parameters
    ----------
    parent          : Parent pointer for each node, from parent_pointer().
    edge_length     : Edge length from each node to its parent, from branch_length().
    leaf_node_index : Traversal index of each leaf, from leaf_arrays().
    """
    dist_from_start = compute_relative_distance(0, parent, edge_length)
    a = int(leaf_node_index[np.argmax(dist_from_start[leaf_node_index])])
    dist_from_a = compute_relative_distance(a, parent, edge_length)
    b = int(leaf_node_index[np.argmax(dist_from_a[leaf_node_index])])
    dist_from_b = compute_relative_distance(b, parent, edge_length)

    diameter = float(dist_from_a[b])
    distance_to_midpoint = np.maximum(dist_from_a[leaf_node_index], dist_from_b[leaf_node_index]) - diameter / 2
    return distance_to_midpoint, diameter
    
# =================================================================
# Integrated function
# =================================================================
@dataclass
class TreeIndex:
    """Indexed tree state produced by build_tree_index()."""
    tree:                   dendropy.Tree
    parent:                 np.ndarray
    edge_length:            np.ndarray
    topological_depth:      np.ndarray
    root_distance:          np.ndarray
    midpoint_root_distance: np.ndarray
    midpoint_diameter:      float
    lift_table:             np.ndarray
    leaf_ids:               np.ndarray
    leaf_node_index:        np.ndarray

def build_tree_index(input_file: Path, schema: str = "newick", order: str = "preorder") -> TreeIndex:
    """
    Loads important tree information.

    Parameters
    ----------
    input_file     : file to load.
    schema         : format [.nwk .json .nexus .phyloxml].
    order          : order of transversal.
    """
    tree  = load(input_file, schema=schema)
    nodes, pointer_parent_node, pointer_parent_edge = create_pointer_parent(tree, order)
    leaf_ids, leaf_node_index = extract_leaves(nodes)
    liftTable = lift_table(pointer_parent_node)
    topological_depth = compute_topological_depth(pointer_parent_node)
    root_distance = compute_rooted_distance(pointer_parent_node, pointer_parent_edge)
    root_to_tip_raw, diameter =diameter_midpoint_distance(pointer_parent_node, pointer_parent_edge, leaf_node_index)
    return TreeIndex(
        tree=                       tree,
        parent=                     pointer_parent_node,
        edge_length=                pointer_parent_edge,
        topological_depth=          topological_depth,
        root_distance=              root_distance,
        midpoint_root_distance=     root_to_tip_raw,
        midpoint_diameter=          diameter,
        lift_table=                 liftTable,
        leaf_ids=                   leaf_ids,
        leaf_node_index=            leaf_node_index,
    )
