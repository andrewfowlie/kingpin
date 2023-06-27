"""
Example with step-functions
===========================
"""

import numpy as np
import matplotlib.pyplot as plt

import kingpin as kp


# Data and points for prediction

x = np.linspace(0, 10, 51)
truth = 100. * np.ones_like(x)
truth[x > 5.05] = 500.
truth[x > 7.55] = 100.
noise = 10 * np.ones_like(x)
y = truth + np.random.randn(*x.shape) * noise
p = np.linspace(x.min(), x.max(), 201)

# Bespoke choices

model = kp.Celerite2(x, y, noise, p)

# Priors and proposals for mean, sigma and length

mean = kp.Uniform(0., 750.)
sigma = kp.Uniform(0., 750.)
length = kp.Uniform(0., 10.)
params = kp.Independent(mean, sigma, length)

# Run RJ-MCMC

rj = kp.TGP(model, params)
rj.walk(n_threads=None, n_iter=1000, n_burn=500)

# Examine results

print(rj.acceptance)
print(rj.arviz_summary())

plt.xlabel('$x$')
plt.ylabel('$y$')
rj.savefig("step_functions_tgp.png")
