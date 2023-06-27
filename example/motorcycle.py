"""
Classic motorcycle data example
===============================
"""

import numpy as np
import matplotlib.pyplot as plt

import kingpin as kp


# Data and points for prediction

x, y = np.loadtxt("motorcycle.txt", unpack=True)
p = np.linspace(x.min(), x.max(), 301)

# Bespoke choices

model = kp.Celerite2(x, y, x_predict=p)

# Priors and proposals for mean, sigma, length and nugget

mean = kp.Uniform(-200., 150.)
sigma = kp.Uniform(0., 500.)
length = kp.Uniform(0., 50.)
nugget = kp.Uniform(0., 300.)
params = kp.Independent(mean, sigma, length, nugget)

# Run RJ-MCMC

rj = kp.TGP(model, params)
rj.walk(thin=10, n_threads=4, n_iter=20000, n_burn=2000, n_iter_params=1, screen=False)

# Examine results

print(rj.acceptance)
print(rj.arviz_summary())

plt.scatter(x, y)
plt.xlabel('Time (ms)')
plt.ylabel('Acceleration ($g$)')
plt.savefig("motorcycle_data.png")

rj.plot()
plt.savefig("motorcycle_tgp.png")
