"""
Unit testing of priors
======================
"""


import unittest
import numpy as np

from kingpin import Normal, Uniform, Independent
from kingpin.prior import CGM
from test_tree import left_tree


class TestPriors(unittest.TestCase):

    def test_normal(self):
        state = np.random.RandomState(1)
        n = Normal(0, 1)
        rvs = n.rvs(state)
        log_pdf_ratio = n.log_pdf_ratio(1, 0)
        self.assertAlmostEqual(rvs, 1.6243453636632417, 6)
        self.assertEqual(log_pdf_ratio, 0.5)

    def test_uniform(self):
        state = np.random.RandomState(1)
        u = Uniform(10, 100)
        rvs = u.rvs(state)
        log_pdf_ratio = u.log_pdf_ratio(50, 60)
        self.assertAlmostEqual(rvs, 47.53198042323166, 6)
        self.assertEqual(log_pdf_ratio, 0.)

    def test_cgm(self):
        cgm = CGM(0.5, 0.75)
        log_pdf_ratio = cgm.log_pdf_ratio(5)
        log_pdf = cgm.log_pdf(left_tree(), 3)
        self.assertAlmostEqual(log_pdf, -4.074563872660783, 6)
        self.assertAlmostEqual(log_pdf_ratio, -2.1442304241708285, 6)

    def test_independent(self):
        state = np.random.RandomState(1)
        indep = Independent(Normal(0, 1), Uniform(10, 100))
        rvs = indep.rvs(state)
        log_pdf_ratio = indep.log_pdf_ratio([1, 50], [0, 60])
        self.assertTrue(np.allclose(rvs, [1.6243453636632417, 10.01029373]))
        self.assertEqual(log_pdf_ratio, 0.5)


if __name__ == '__main__':
    unittest.main()
