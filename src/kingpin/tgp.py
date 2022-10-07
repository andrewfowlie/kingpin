"""
Thread multiple RJ-MCMC chains
==============================
"""

import multiprocessing
from typing import Optional

import arviz
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np

from .model import Celerite2, Model
from . import plot
from .prior import Independent, Uniform, Prior
from .rjmcmc import RJMCMC
from .recorder import Recorder


class TGP:
    """
    RJ-MCMC method applied on treed Gaussian processes on multiple threads
    """

    def __init__(self,
                 model: Model,
                 params_prior: Prior,
                 systematic_prior: Optional[Prior] = None,
                 seed: Optional[int] = None,
                 alpha: Optional[float] = 0.5,
                 beta: Optional[float] = 2.,
                 change_scale_factor: Optional[float] = 2.,
                 change_param_factor: Optional[float] = 0.05):
        """
        :param model: Gaussian process model
        :param params_prior: Prior for Gaussian process model parameters
        :param systematic_prior: Prior for any systematic parameters
        :param seed: Seed for reproducible result
        :param alpha: Parameter in CGM prior for tree
        :param beta: Parameter in CGM prior for tree
        :param change_scale_factor: Scale of proposal for splitting rule
        :param change_param_factor: Scale of proposal for parameters
        """
        self.model = model
        self.rjmcmc_args = [model, params_prior]
        self.rjmcmc_kwargs = dict(systematic_prior=systematic_prior,
                                  alpha=alpha,
                                  beta=beta,
                                  change_param_factor=change_param_factor,
                                  change_scale_factor=change_scale_factor)
        self.seed_sequence = np.random.SeedSequence(seed)
        self.threads = None

    def make_thread(self, seed):
        """
        Make an RJMCM instance for a thread
        """
        return RJMCMC(*self.rjmcmc_args, seed=seed, **self.rjmcmc_kwargs)

    @classmethod
    def from_data(cls,
                  x_data: np.typing.ArrayLike,
                  y_data: np.typing.ArrayLike,
                  noise: Optional[np.typing.ArrayLike] = None,
                  x_predict: Optional[np.typing.ArrayLike] = None,
                  **kwargs):
        """
        Interface that makes generic modeling choices from data
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

    def walk(self, num_cores=None, **kwargs) -> None:
        """
        Multiple RJ-MCMC walks
        """
        num_cores = num_cores if num_cores else multiprocessing.cpu_count()
        seeds = self.seed_sequence.spawn(num_cores)

        if num_cores == 1:
            self.threads = [self.thread(seeds[0], **kwargs)]
        else:
            self.threads = Parallel(n_jobs=num_cores)(
                delayed(self.thread)(
                    seed, screen=False, position=i, **kwargs)
                for i, seed in enumerate(seeds))

    @property
    def acceptance(self):
        """
        :return: Aggregate acceptance information
        """
        return sum((t.acceptance for t in self.threads), Recorder())

    @property
    def arviz(self):
        """
        :return: Summary of chains including ESS and R-hat
        """
        summaries = np.stack([np.array(t.summaries) for t in self.threads])
        return arviz.summary(arviz.convert_to_dataset(summaries))

    @property
    def mean(self):
        """
        :return: Aggregated model prediction for mean
        """
        return np.mean([t.mean for t in self.threads], axis=0)

    @property
    def stdev(self):
        """
        :return: Aggregated model prediction for standard deviation
        """
        return np.diag(self.cov)**0.5

    @property
    def cov(self):
        """
        :return: Aggregated model prediction for covariance
        """
        second_moments = np.mean(
            [t.second_moments for t in self.threads], axis=0)
        return second_moments - np.outer(self.mean, self.mean)

    @property
    def edge_counts(self):
        """
        :return: Aggregated edge counts
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
        """
        self.plot(*args, **kwargs)
        plt.savefig(filename)

    def savetxt(self, filename: str) -> None:
        """
        Saves predictions to disk
        """
        np.savetxt(filename, np.array(
            [self.mean, self.stdev]).T, header="mean st.dev")
