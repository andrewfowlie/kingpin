"""
Thread multiple RJ-MCMC chains
==============================
"""

from __future__ import annotations

import multiprocessing
from typing import Optional

import arviz
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np

from .model import Celerite2, Model
from . import plot
from .prior import Independent, Uniform, Prior, TreePrior, CGM
from .proposal import Proposal, TruncatedProposal, FractionalProposal
from .rjmcmc import RJMCMC
from .recorder import Recorder
from .alias import ArrayLike


class TGP:
    """
    RJ-MCMC method applied on treed Gaussian processes on multiple threads
    """

    def __init__(self,
                 model: Model,
                 params_prior: Prior,
                 systematic_prior: Optional[Prior] = None,
                 seed: Optional[int] = None,
                 tree_prior: Optional[TreePrior] = CGM(0.5, 2),
                 params_proposal: Optional[Proposal] = FractionalProposal(0.05),
                 systematic_proposal: Optional[Proposal] = FractionalProposal(0.05),
                 change_proposal: Optional[TruncatedProposal] = None):
        """
        :param model: Gaussian process model
        :param params_prior: Prior for leaf parameters
        :param systematic_prior: Prior for systematic parameters
        :param seed: Seed for reproducible result
        :param tree_prior: Prior for tree
        :param params_proposal: Proposal for leaf parameters
        :param systematic_proposal: Proposal for systematic parameters
        :param change_proposal: Proposal for changing split parameters
        """
        self.model = model
        self.rjmcmc_args = [model, params_prior]
        self.rjmcmc_kwargs = dict(systematic_prior=systematic_prior,
                                  tree_prior=tree_prior,
                                  params_proposal=params_proposal,
                                  systematic_proposal=systematic_proposal,
                                  change_proposal=change_proposal)
        self.seed_sequence = np.random.SeedSequence(seed)
        self.threads = None

    def make_thread(self, seed):
        """
        Make an RJMCM instance for a thread
        """
        return RJMCMC(*self.rjmcmc_args, seed=seed, **self.rjmcmc_kwargs)

    @classmethod
    def from_data(cls,
                  x_data: ArrayLike,
                  y_data: ArrayLike,
                  noise: Optional[ArrayLike] = None,
                  x_predict: Optional[ArrayLike] = None,
                  **kwargs):
        """
        Interface that makes generic modeling choices from data

        :param x_data: Input locations
        :param y_data: Measurements
        :param noise: Diagonal measurement error
        :param x_predict: Locations of predictions
        """
        model = Celerite2(x_data, y_data, noise, x_predict)

        # Priors and proposals for mean, sigma and length

        mean = Uniform(y_data.min() - np.abs(y_data.min()),
                       y_data.max() + np.abs(y_data.max()))
        sigma = Uniform(0., np.abs(y_data.max() - y_data.min()))
        length = Uniform(0., np.abs(x_data.max() - x_data.min()))

        # Priors for nugget term

        if noise is None:
            nugget = Uniform(0., np.abs(y_data.max() - y_data.min()))
            params = Independent(mean, sigma, length, nugget)
        else:
            params = Independent(mean, sigma, length)

        # Build model

        return cls(model, params, **kwargs)

    def thread(self, seed, *args, **kwargs):
        """
        Construct and walk RJ-MCMC instance
        """
        thread = self.make_thread(seed)
        thread.walk(*args, **kwargs)
        return thread

    def walk(self, n_threads: Optional[int] = None, screen: Optional[bool] = False, **kwargs) -> None:
        """
        Multiple RJ-MCMC walks

        :param n_threads: Number of threads
        :param screen: Show detailed state of tree on screen
        """
        n_threads = n_threads if n_threads else multiprocessing.cpu_count()
        seeds = self.seed_sequence.spawn(n_threads)

        if n_threads == 1:
            self.threads = [self.thread(seeds[0], screen=screen, **kwargs)]
        else:
            self.threads = Parallel(n_jobs=n_threads)(
                delayed(self.thread)(
                    seed, screen=False, position=i, **kwargs)
                for i, seed in enumerate(seeds))

    @property
    def acceptance(self):
        """
        Aggregated acceptance information
        """
        return sum((t.acceptance for t in self.threads), Recorder())

    def to_arviz(self):
        """
        :return: Summary data in arviz format
        """
        summaries = np.stack([np.array(t.summaries) for t in self.threads])
        return arviz.convert_to_dataset(summaries)

    def arviz_summary(self):
        """
        :return: Summary of chains including ESS and R-hat
        """
        return arviz.summary(self.to_arviz())

    @property
    def mean(self):
        """
        Aggregated model prediction for mean
        """
        return np.mean([t.mean for t in self.threads], axis=0)

    @property
    def stdev(self):
        """
        Aggregated model prediction for standard deviation
        """
        return np.diag(self.cov)**0.5

    @property
    def cov(self):
        """
        Aggregated model prediction for covariance
        """
        second_moments = np.mean(
            [t.second_moments for t in self.threads], axis=0)
        return second_moments - np.outer(self.mean, self.mean)

    @property
    def edge_counts(self):
        """
        Aggregated edge counts
        """
        return np.sum([t.edge_counts for t in self.threads], axis=0)

    def plot(self, **kwargs) -> None:
        """
        Plot of aggregated predictions
        """
        plot.data(self.model.x_data, self.model.y_data, self.model.noise, **kwargs)
        plot.pred(self.model.x_predict, self.mean, self.stdev, **kwargs)
        plot.edges(self.model.x_data, self.edge_counts, **kwargs)

    def show(self, *args, **kwargs) -> None:
        """
        Show plot of predictions
        """
        self.plot(*args, **kwargs)
        plt.show()

    def savefig(self, filename: str, *args, **kwargs) -> None:
        """
        Save plot of predictions

        :param filename: File name for figure
        """
        self.plot(*args, **kwargs)
        plt.savefig(filename)

    def savetxt(self, filename: str) -> None:
        """
        Saves mean and standard deviations to disk

        :param filename: File name for text file of data
        """
        np.savetxt(filename, np.array(
            [self.mean, self.stdev]).T, header="mean st.dev")
