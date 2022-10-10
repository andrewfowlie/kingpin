"""
Represent separable priors or proposal
======================================
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import typing, log1p


class CGM:
    """
    CGM prior for tree structure
    """

    def __init__(self, alpha, beta):
        """
        :return: Log of splitting probability
        """
        self.alpha = alpha
        self.beta = beta
        self.log_alpha = np.log(self.alpha)

    def log_pdf(self, subtree, subtree_depth):
        """
        :return: Log of splitting probability
        """
        log_p = 0.

        for relative_depth, level in enumerate(subtree.levels):
            depth = relative_depth + subtree_depth
            log_p_leaf = log1p(-self.alpha * (1. + depth)**-self.beta)
            log_p_parent = self.log_alpha - self.beta * log1p(depth)

            for node in level:
                if node.is_leaf():
                    log_p += log_p_leaf
                else:
                    log_p += log_p_parent

        return log_p

    def log_pdf_ratio(self, depth):
        """
        :return: Contribution to acceptance probability for grow
        """
        log_p = self.log_alpha - self.beta * np.log1p(depth)
        log_q1 = np.log1p(-self.alpha * (depth + 1)**-self.beta)
        log_q2 = np.log1p(-self.alpha * (depth + 2)**-self.beta)
        return log_p - log_q1 + 2. * log_q2


class Prior(ABC):
    """
    Represent prior for model's parameters
    """
    @abstractmethod
    def rvs(self, random_state, *args, **kwargs) -> typing.ArrayLike:
        """
        :return: Draw from prior
        """

    @abstractmethod
    def log_pdf_ratio(self, params: typing.ArrayLike, other: typing.ArrayLike) -> float:
        """
        :return: Log pdf ratio of prior
        """


class Uniform(Prior):
    """
    Performant implementation of uniform distribution
    """

    def __init__(self, lower, upper):
        """
        Interval for distribution
        """
        self.lower, self.upper = lower, upper

    def rvs(self, random_state, *args, **kwargs):
        """
        :return: Random draw from distribution
        """
        return random_state.uniform(self.lower, self.upper, *args, **kwargs)

    def log_pdf_ratio(self, params, other):
        """
        :return: Trivial constant
        """
        if np.any(other < self.lower) or np.any(other > self.upper):
            return -np.inf
        return 0.


class Normal(Prior):
    """
    Performant implementation of normal distribution
    """

    def __init__(self, mean, sigma):
        """
        Interval for distribution
        """
        self.mean, self.sigma = mean, sigma

    def rvs(self, random_state, *args, **kwargs):
        """
        :return: Random draw from distribution
        """
        return random_state.normal(self.mean, self.sigma, *args, **kwargs)

    def log_pdf_ratio(self, params, other):
        """
        :return: Log pdf ratio for normal distribution
        """
        delta = (other - self.mean)**2 - (params - self.mean)**2
        return -0.5 * np.sum(delta / self.sigma**2)


class Proposal:
    """
    Fractional proposal
    """

    def __init__(self, frac):
        """
        :param frac: Fractional change in parameter
        """
        self.frac = frac

    def rvs(self, params, random_state):
        """
        Random draw from uniform centered at current parameters
        """
        delta = np.abs(self.frac * params)
        return params + random_state.uniform(-delta, delta)

    @staticmethod
    def log_pdf_ratio(params, other):
        """
        Log pdf ratio for fractional proposal
        """
        return np.log(np.abs(params) / np.abs(other)).sum()


class Independent(Prior):
    """
    Prior for independent parameters
    """

    def __init__(self, *prior):
        """
        :param *params: Pairs of prior and proposal distributions
        """
        self._log_pdf_ratio = [p.log_pdf_ratio for p in prior]
        self._rvs = [p.rvs for p in prior]

    def rvs(self, random_state, *args, **kwargs):
        """
        :return: Draw from each parameter's prior
        """
        return np.array([r(random_state=random_state, *args, **kwargs) for r in self._rvs])

    def log_pdf_ratio(self, params, other):
        """
        :return: Sum of log pdf ratio for each parameter
        """
        return sum(r(p, q) for p, q, r in zip(params, other, self._log_pdf_ratio))
