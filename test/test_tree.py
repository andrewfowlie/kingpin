"""
Unit testing of tree
====================
"""


import unittest
import numpy as np

from kingpin.tree import Tree


PARAMS = np.ones(3)


def left_tree():
    tree = Tree(PARAMS)
    tree.grow(0.5, PARAMS, lhs=True)
    tree.left.grow(0.25, PARAMS, lhs=True)
    return tree


def right_tree():
    tree = Tree(PARAMS)
    tree = Tree(PARAMS)
    tree.grow(0.25, PARAMS, lhs=True)
    tree.right.grow(0.5, PARAMS, lhs=True)
    return tree


STR = r"""3 leaves

0.00 - 0.25 : [1. 1. 1.] | 0.25 - 0.50 : [1. 1. 1.] | 0.50 - 1.00 : [1. 1. 1.]


               ______________0.5_____
              /                      \
       _____0.25_____            no children
      /              \
no children      no children
"""


class TestTree(unittest.TestCase):

    def test_rotate_lhs(self):
        tree = right_tree()
        self.assertTrue(tree.is_rotatable(lhs=True))
        self.assertFalse(tree.is_rotatable(lhs=False))
        tree.rotate(lhs=True)
        self.assertFalse(tree.equal_structure(right_tree()))
        self.assertTrue(tree.equal_structure(left_tree()))

    def test_rotate_rhs(self):
        tree = left_tree()
        self.assertTrue(tree.is_rotatable(lhs=False))
        self.assertFalse(tree.is_rotatable(lhs=True))
        tree.rotate(lhs=False)
        self.assertFalse(tree.equal_structure(left_tree()))
        self.assertTrue(tree.equal_structure(right_tree()))

    def test_grow_prune(self):
        tree = left_tree()
        tree.right.grow(0.5, PARAMS, True)
        tree.left.prune(True)
        tree.value = 0.25
        self.assertFalse(tree.equal_structure(left_tree()))
        self.assertTrue(tree.equal_structure(right_tree()))

    def test_depth(self):
        tree = left_tree()
        self.assertTrue(tree.get_depth(tree.right) == 1)
        self.assertTrue(tree.get_depth(tree.left.right) == 2)
        self.assertTrue(tree.get_depth(tree.left.left) == 2)

    def test_parents(self):
        tree = left_tree()
        self.assertEqual(len(tree.parents()), 2)

    def test_is_leaf(self):
        tree = left_tree()
        self.assertFalse(tree.is_leaf())
        self.assertTrue(tree.left.right.is_leaf())

    def test_interval(self):
        tree = left_tree()
        self.assertEqual(tree.interval(tree.left), (0., 0.5))
        self.assertEqual(tree.interval(tree.right), (0.5, 1.))
        self.assertEqual(tree.interval(tree.left.left), (0., 0.25))
        self.assertEqual(tree.interval(tree.left.right), (0.25, 0.5))

    def test_str(self):
        str_ = str(left_tree())
        self.assertEqual(str_, STR)


if __name__ == '__main__':
    unittest.main()
