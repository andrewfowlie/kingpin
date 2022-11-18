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

# Bespoke choices

model = kp.Celerite2(x, y, x_predict=p)

# Priors and proposals for mean, sigma, length and nugget

mean = kp.Uniform(0., 150.)
sigma = kp.Uniform(0., 500.)
length = kp.Uniform(0., 50.)
nugget = kp.Uniform(0., 300.)
params = kp.Independent(mean, sigma, length, nugget)

# Run RJ-MCMC

rj = kp.TGP(model, params)
rj.walk(thin=2, n_cores=1, n_iter=22000, n_burn=20000, n_iter_params=1, screen=False)

print(rj.acceptance)
print(rj.arviz_summary())
rj.show()
