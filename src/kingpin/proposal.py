"""
Represent proposals
===================
"""

from abc import ABC, abstractmethod
import numpy as np

from .draw import truncnorm


class Proposal(ABC):
    """
    Proposal distribution for e.g. model parameters
    """
    @abstractmethod
    def rvs(self, params, random_state):
        """
        Random draw from proposal
        """

    @abstractmethod
    def log_pdf_ratio(self, params, other):
        """
        Log pdf ratio for proposal
        """


class FractionalProposal(Proposal):
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


class GaussianProposal(Proposal):
    """
    Gaussian proposal
    """

    def __init__(self, scale):
        """
        :param scale: Scale parameter for Gaussian
        """
        self.scale = scale

    def rvs(self, params, random_state):
        """
        Random draw from Gaussian centered at current parameters
        """
        return random_state.normal(params, self.scale)

    @staticmethod
    def log_pdf_ratio(params, other):
        """
        Log pdf ratio for Gaussian proposal
        """
        return 0.


class TruncatedProposal(ABC):
    """
    Truncated proposal distribution for e.g. splitting rule
    """
    @abstractmethod
    def rvs(self, params, random_state, lower, upper):
        """
        Random draw from proposal
        """

    @abstractmethod
    def log_pdf_ratio(self, params, other, lower, upper):
        """
        Log pdf ratio for proposal
        """


class TruncatedGaussianProposal(TruncatedProposal):
    """
    Truncated Gaussian proposal
    """

    def __init__(self, scale):
        """
        :param scale: Scale parameter for Gaussian
        """
        self.scale = scale

    def rvs(self, params, random_state, lower, upper):
        """
        Random draw from Gaussian centered at current parameters and truncated
        """
        return truncnorm(lower, upper, params, self.scale, random_state)

    @staticmethod
    def log_pdf_ratio(params, other, lower, upper):
        """
        Log pdf ratio for Gaussian proposal
        """
        return 0.
