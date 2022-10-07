"""
Random sampling
===============
"""


import numpy as np
# pylint: disable-next=no-name-in-module
from scipy.special import ndtri, ndtr


def truncnorm(lower, upper, mean, sigma, random_state):
    """
    :return: Truncated normal draw
    """
    limits = np.array([lower, upper])
    standard_normal_limits = (limits - mean) / sigma
    uniform_limits = ndtr(standard_normal_limits)
    uniform_draw = random_state.uniform(*uniform_limits)
    standard_normal_draw = ndtri(uniform_draw)
    return sigma * standard_normal_draw + mean


def bernoulli(random_state):
    """
    :return: Bernoulli random draw
    """
    return random_state.integers(2).astype(bool)


def choice(lst, random_state):
    """
    :return: Random draw from list
    """
    if lst:
        idx = int(random_state.integers(len(lst)))
        return lst[idx]

    return None
