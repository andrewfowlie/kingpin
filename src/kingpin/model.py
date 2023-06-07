"""
Model for RJ-MCMC Treed-GP
==========================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional
import numpy as np

import celerite2

from .alias import ArrayLike


class Model(ABC):
    """
    Gaussian process model of data
    """

    def __init__(self, x_data: ArrayLike,
                 y_data: ArrayLike,
                 noise: Optional[ArrayLike] = None,
                 x_predict: Optional[ArrayLike] = None):
        """
        :param x_data: Input locations
        :param y_data: Measurements
        :param noise: Diagonal measurement error
        :param x_predict: Locations of predictions
        """
        self.x_data = x_data
        self.x_data_min = x_data.min()
        self.x_data_delta = self.x_data.max() - self.x_data.min()
        self.x_data_scaled = (self.x_data - self.x_data_min) / self.x_data_delta
        self.y_data = y_data
        self.x_predict = x_predict if x_predict is not None else x_data
        self.noise = noise if noise is not None else np.zeros_like(x_data)

    def where(self, arr, interval):
        """
        :return: Partitions according to an interval
        """
        if interval[0] <= 0.:
            lower = -np.inf
        else:
            lower = interval[0] * self.x_data_delta + self.x_data_min

        if interval[1] >= 1.:
            upper = np.inf
        else:
            upper = interval[1] * self.x_data_delta + self.x_data_min

        return (arr <= upper) & (arr >= lower)

    def min_data_points(self, intervals):
        """
        :return: Minimim number of data points inside partitions
        """
        return min(self.where(self.x_data, i).sum() for i in intervals)

    @abstractmethod
    def loglike_leaf(self, interval: ArrayLike,
                     params: ArrayLike,
                     systematic: ArrayLike):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Log probability for GP on one leaf
        """

    @abstractmethod
    def pred_leaf(self, interval: ArrayLike,
                  params: ArrayLike,
                  systematic: ArrayLike):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Mean and covariance for GP on one leaf
        """

    def pred(self, intervals, params, systematic):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Mean and covariance summed over all partitions
        """
        mean = 0.
        cov = 0.

        if systematic is not None:
            systematic = tuple(systematic)

        for interval, param in zip(intervals, params):
            mean_, cov_ = self.pred_leaf(interval, tuple(param), systematic)
            mean += mean_
            cov += cov_

        return mean, cov

    def loglike(self, intervals, params, systematic):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Log likelihood summed over all partitions
        """
        if systematic is not None:
            systematic = tuple(systematic)
        return sum(self.loglike_leaf(interval, tuple(p), systematic)
                   for interval, p in zip(intervals, params))

    @staticmethod
    @abstractmethod
    def summary(mean, cov):
        """
        :return: Summaries to save and assess
        """


class Celerite2(Model):
    """
    Gaussian process model using celerite2 library
    """
    @lru_cache(maxsize=1)
    def gp_leaf(self, interval, params, systematic):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Log probability, mean and coviance for GP on one leaf
        """
        where = self.where(self.x_data, interval)

        if len(params) == 3:
            mean, sigma, length = params
            nugget = 0
        else:
            mean, sigma, length, nugget = params

        kernel = celerite2.terms.Matern32Term(
            sigma=np.abs(sigma), rho=np.abs(length))
        gp_leaf = celerite2.GaussianProcess(kernel, mean=mean)
        gp_leaf.compute(self.x_data[where], yerr=self.noise[where] + np.abs(nugget))
        return gp_leaf

    @lru_cache(maxsize=32)
    def pred_leaf(self, interval, params, systematic):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Log probability, mean and coviance for GP on one leaf
        """
        gp_leaf = self.gp_leaf(interval, params, systematic)

        y_data = self.y_data[self.where(self.x_data, interval)]
        where = self.where(self.x_predict, interval)

        mean, cov = gp_leaf.predict(
            y_data, t=self.x_predict[where], return_cov=True)

        if len(params) != 3:
            nugget = params[3]
            cov += nugget**2 * np.identity(cov.shape[0])

        padded_mean = np.zeros(where.shape)
        padded_mean[where] = mean

        padded_cov = np.zeros((where.shape[0], where.shape[0]))
        padded_cov[np.outer(where, where)] = cov.flatten()

        return padded_mean, padded_cov

    @lru_cache(maxsize=32)
    def loglike_leaf(self, interval, params, systematic):
        """
        :param interval: Interval of this leaf
        :param params: Parameters for this leaf
        :param systematic: Systematic parameters common to all leaves

        :return: Log probability, mean and coviance for GP on one leaf
        """
        y_data = self.y_data[self.where(self.x_data, interval)]
        try:
            return self.gp_leaf(interval, params, systematic).log_likelihood(y_data)
        # pylint: disable-next=c-extension-no-member
        except celerite2.driver.LinAlgError:
            return -np.inf

    @staticmethod
    def summary(mean, cov):
        """
        :return: Mean over all predictions
        """
        return np.mean(mean)
