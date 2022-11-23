"""
Unit testing of model
=====================
"""


import unittest
import numpy as np

from kingpin import Celerite2


np.random.seed(1)
x = np.linspace(0, 100, 100)
y = np.random.randn(100) * 5 + 10
c2 = Celerite2(x, y)
systematic = None
intervals = [(0., 0.75), (0.75, 1.)]
params = [(5., 5., 5., 5.), (10., 10., 10., 10.)]


class TestCelerite2(unittest.TestCase):

    def test_loglike(self):
        loglike = c2.loglike(intervals, params, systematic)
        self.assertAlmostEqual(loglike, -320.70791550191075, 6)

    def test_pred(self):
        mean, cov = c2.pred(intervals, params, systematic)
        self.assertAlmostEqual(mean[50], 10.063686391528414, 6)
        self.assertAlmostEqual(cov[20, 21], 3.999796214986958, 6)


if __name__ == '__main__':
    unittest.main()
