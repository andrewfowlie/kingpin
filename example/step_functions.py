"""
Example of running RJ-MCMC
==========================
"""

import numpy as np
import kingpin as kp


# Data and points for prediction

x = np.linspace(0, 10, 101)
y = 100. * np.ones_like(x)
y[x > 5.05] = 300.
y[x > 7.55] = 100.
noise = np.ones_like(x)
p = np.linspace(x.min(), x.max(), 201)

# Bespoke choices

model = kp.Celerite2(x, y, noise, p)

# Priors and proposals for mean, sigma and length

mean = kp.Uniform(0., 500.)
sigma = kp.Uniform(0., 500.)
length = kp.Uniform(0., 10.)
params = kp.Independent(mean, sigma, length)

# Run RJ-MCMC

rj = kp.TGP(model, params)
rj.walk(n_threads=1, n_iter=1000, n_burn=500)

print(rj.acceptance)
print(rj.arviz_summary())
rj.show()
