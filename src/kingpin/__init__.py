"""
Treed Gaussian Processes with RJ-MCMC marginalization
=====================================================

This a Python implementation of the treed Gaussian process algorithm.
"""

from .rjmcmc import RJMCMC
from .model import Model, Celerite2
from .prior import Normal, Uniform, Prior, Independent, CGM, TreePrior
from .proposal import GaussianProposal, FractionalProposal
from .tgp import TGP
