"""
Unit testing of TGP
===================
"""


import unittest
import numpy as np

import kingpin as kp


# Walk a TGP on example data

x = np.linspace(0, 10, 101)
y = x**2
noise = np.ones_like(x)
p = np.linspace(x.min(), x.max(), 101)

model = kp.Celerite2(x, y, noise, p)

mean = kp.Uniform(0., 500.)
sigma = kp.Uniform(0., 500.)
length = kp.Uniform(0., 10.)
params = kp.Independent(mean, sigma, length)

rj = kp.TGP(model, params, seed=1)
rj.walk(n_iter=100, n_burn=10, n_cores=1)


class TestCelerite2(unittest.TestCase):

    def test_mean(self):
        self.assertAlmostEqual(rj.mean[50], 25.00069809839112, 6)

    def test_stdev(self):
        self.assertAlmostEqual(rj.stdev[50], 0.38795539435550014, 6)

    def test_edge_counts(self):
        self.assertEqual(rj.edge_counts.sum(), 0)

    def test_arviz(self):
        res = rj.arviz_summary()["mean"][0]
        self.assertAlmostEqual(res, 33.5, 3)

    def test_from_data(self):
        fd = kp.TGP.from_data(x, y, seed=1)
        fd.walk(n_iter=100, n_burn=10, n_cores=1)
        self.assertAlmostEqual(fd.mean[50], 25.023031491400552, 6)

if __name__ == '__main__':
    unittest.main()
