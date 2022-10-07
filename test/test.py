"""
Unit testing
============
"""


import unittest
import numpy as np

from kingpin.tree import Tree, VALUE_NONE


PARAMS = np.ones(3)


def left_tree():
    tree = Tree(PARAMS)
    tree.left = Tree(PARAMS)
    tree.right = Tree(PARAMS)
    tree.left.left = Tree(PARAMS)
    tree.left.right = Tree(PARAMS)
    return tree

def right_tree():
    tree = Tree(PARAMS)
    tree.left = Tree(PARAMS)
    tree.right = Tree(PARAMS)
    tree.right.left = Tree(PARAMS)
    tree.right.right = Tree(PARAMS)
    return tree


class TestTree(unittest.TestCase):

    def test_rotate_lhs(self):
        tree = right_tree()
        tree.rotate(lhs=True)
        self.assertFalse(tree.equal_structure(right_tree()))
        self.assertTrue(tree.equal_structure(left_tree()))

    def test_rotate_rhs(self):
        tree = left_tree()
        tree.rotate(lhs=False)
        self.assertFalse(tree.equal_structure(left_tree()))
        self.assertTrue(tree.equal_structure(right_tree()))

    def test_grow_prune(self):
        tree = left_tree()
        tree.right.grow(VALUE_NONE, PARAMS, True)
        tree.left.prune(True)
        self.assertFalse(tree.equal_structure(left_tree()))
        self.assertTrue(tree.equal_structure(right_tree()))

    def test_depth(self):
        tree = left_tree()
        self.assertTrue(tree.get_depth(tree.right) == 1)
        self.assertTrue(tree.get_depth(tree.left.right) == 2)
        self.assertTrue(tree.get_depth(tree.left.left) == 2)

if __name__ == '__main__':
    unittest.main()
