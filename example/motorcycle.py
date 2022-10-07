"""
Example of running RJ-MCMC
==========================
"""

import numpy as np
import kingpin as kp

# Data and points for prediction

x, y = np.loadtxt("motorcycle.txt", unpack=True)
y -= y.min()
p = np.linspace(x.min(), x.max(), 301)

# Simple interface

#rj = kp.kingpin(x, y, p=p)
#rj.walk()
#rj.show()

# Bespoke choices

model = kp.Celerite2(x, y, x_predict=p)

# Priors and proposals for mean, sigma and length
mean = kp.Uniform(0., 150.)
sigma = kp.Uniform(0., 500.)
length = kp.Uniform(0., 50.)
nugget = kp.Uniform(0., 300.)
params = kp.Independent(mean, sigma, length, nugget)

# Run RJ-MCMC

rj = kp.TGP(model, params)
rj.walk(thin=2, num_cores=1, n_iter=22000, n_burn=2000, n_iter_params=1, screen=False)
rj.show()
print(rj.acceptance)
print(rj.arviz)
