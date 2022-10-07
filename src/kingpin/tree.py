"""
Represent a partition tree
==========================
"""

import copy
import numpy as np
from numpy import array2string
from binarytree import Node, get_parent


class ValueNone(float):
    """
    Float type that stringifies to something desirable
    """

    def __str__(self):
        return "no children"


VALUE_NONE = ValueNone(np.inf)


class Tree(Node):
    """
    Represent a tree. Each node has associated parameters. Operations
    motivated by RJ-MCMC move proposals
    """

    def __init__(self, node_params, value=None):
        self.node_params = node_params
        super().__init__(value=VALUE_NONE)

    def is_leaf(self):
        """
        :return: Whether node is a leaf
        """
        return self.left is None and self.right is None

    def parents(self):
        """
        :return: List of parents
        """
        return [node for node in self if not node.is_leaf()]

    def prunable(self):
        """
        :return: List of prunable parents
        """
        return [node for node in self
                if not node.is_leaf() and node.left.is_leaf() and node.right.is_leaf()]

    def is_rotatable(self, lhs):
        """
        :return: Whether node is rotatable
        """
        return ((lhs and self.right is not None and self.right.right is not None)
                or (not lhs and self.left is not None and self.left.left is not None))

    def get_depth(self, node):
        """
        :return: Depth of node relative to root
        """
        for depth, nodes in enumerate(self.levels):
            if node in nodes:
                return depth
        raise RuntimeError("node not in tree")

    def edges(self):
        """
        :return: Edges of boundaries in tree
        """
        return np.sort([0., 1.] + [v for v in self.values if v is not None and not np.isinf(v)])

    def interval(self, node):
        """
        :return: Interval for a particular node
        """

        # leaf node

        if node.is_leaf():

            parent = get_parent(self, node)

            if parent is None:
                return (0., 1.)

            edges = self.edges()

            if parent.left == node:
                upper = parent.value
                lower = edges[edges < upper].max()
            elif parent.right == node:
                lower = parent.value
                upper = edges[edges > lower].min()

            return (lower, upper)

        # regular node

        edges = self.edges()
        lower = edges[edges < node.value].max()
        upper = edges[edges > node.value].min()

        return (lower, upper)

    def params(self):
        """
        :return: Parameters associated with leaves
        """
        return (node.node_params for node in self.inorder if node.is_leaf())

    def intervals(self):
        """
        :return: Flatten tree to parameters at leaves
        """
        edges = self.edges()
        return tuple(zip(edges, edges[1:]))

    def __str__(self):
        """
        Modify string of tree to include intervals
        """
        tree = super().__str__()
        leaves = [f"{i[0]:.2f} - {i[1]:.2f} : {array2string(p, precision=2)}"
                  for p, i in zip(self.params(), self.intervals())]
        return f"{self.leaf_count} leaves\n\n" + " | ".join(leaves) + "\n\n" + tree

    def equal_structure(self, other) -> bool:
        """
        :return: Whether tree structures are equal
        """
        return np.all(self.values == other.values)

    def rotate(self, lhs=True):
        """
        Rotate this node either left or right
        """
        value = self.value
        node_params = self.node_params

        if lhs:

            self.value = self.right.value
            self.node_params = self.right.node_params

            self.left.left = copy.copy(self.left)
            self.left.right = self.right.left
            self.right = self.right.right

            self.left.value = value
            self.left.node_params = node_params

        else:

            self.value = self.left.value
            self.node_params = self.left.node_params

            self.right.right = copy.copy(self.right)
            self.right.left = self.left.right
            self.left = self.left.left

            self.right.value = value
            self.right.node_params = node_params

    def grow(self, value, node_params, lhs):
        """
        Grow this node
        """
        self.value = value
        self.left = Tree(self.node_params) if lhs else Tree(node_params)
        self.right = Tree(node_params) if lhs else Tree(self.node_params)

    def prune(self, lhs):
        """
        Prune this node
        """
        self.node_params = self.left.node_params if lhs else self.right.node_params
        self.value = VALUE_NONE
        self.left = None
        self.right = None
