"""
Unit testing of proposals
=========================
"""


import unittest
import numpy as np

from kingpin import FractionalProposal, GaussianProposal


class TestPriors(unittest.TestCase):

    def test_fractional(self):
        state = np.random.RandomState(1)
        f = FractionalProposal(0.05)
        rvs = f.rvs(np.array([10, 10]), state)
        log_pdf_ratio = f.log_pdf_ratio(1, 2)
        self.assertTrue(np.allclose(rvs, np.array([9.917022, 10.22032449])))
        self.assertEqual(log_pdf_ratio, np.log(0.5))

    def test_gaussian(self):
        state = np.random.RandomState(1)
        g = GaussianProposal(0.05)
        rvs = g.rvs(np.array([10, 10]), state)
        log_pdf_ratio = g.log_pdf_ratio(1, 0)
        self.assertTrue(np.allclose(rvs, np.array([10.08121727, 9.96941218])))
        self.assertEqual(log_pdf_ratio, 0.)


if __name__ == '__main__':
    unittest.main()
