"""
RJ-MCMC implementation using the GP model and tree
==================================================
"""

from contextlib import nullcontext
from typing import Optional

import numpy as np
from tqdm import tqdm

from . import draw
from .prior import TreePrior, CGM, Prior
from .proposal import Proposal, TruncatedProposal, FractionalProposal, TruncatedGaussianProposal
from .tree import Tree
from .stack import Stack, get_screen_size
from .recorder import Recorder
from .model import Model


class RJMCMC:
    """
    Recursive-Jump MCMC on a treed Gaussian process
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
        self.params_prior = params_prior
        self.systematic_prior = systematic_prior
        self.random_state = np.random.default_rng(seed)
        self.tree_prior = tree_prior
        self.params_proposal = params_proposal
        self.systematic_proposal = systematic_proposal
        self.change_proposal = change_proposal

        if self.change_proposal is None:
            self.change_proposal = TruncatedGaussianProposal(2. / len(self.model.x_data))

        self.acceptance = Recorder()

        self.loglike = self.tree = self.systematic = None
        self.set_start()
        self.set_accumalators()

    def set_start(self):
        """
        Set start of chain
        """
        params = self.params_prior.rvs(self.random_state)
        self.tree = Tree(params)

        if self.systematic_prior is not None:
            self.systematic = self.systematic_prior.rvs(
                self.random_state)
        else:
            self.systematic = None

        self.loglike = self.model.loglike(
            self.tree.intervals(), self.tree.params(), self.systematic)

    def set_accumalators(self):
        """
        Start accumulators from scratch
        """
        self.n_samples = 0
        self.summaries = []
        self.mean = 0.
        self.second_moments = 0.
        self.edge_counts = 0

    def accumulate(self):
        """
        Accumulate quantities from chain
        """
        mean, cov = self.model.pred(
            self.tree.intervals(), self.tree.params(), self.systematic)
        edge_counts = np.histogram(
            self.tree.edges()[1:-1], bins=self.model.x_data_scaled)[0]
        summary = self.model.summary(mean, cov)

        self.n_samples += 1
        self.summaries.append(summary)
        self.edge_counts += edge_counts
        self.mean += (mean - self.mean) / self.n_samples
        outer = np.outer(mean, mean)
        self.second_moments += (outer + cov -
                                self.second_moments) / self.n_samples

    def walk(self, n_iter: Optional[int] = 5000,
             n_burn: Optional[int] = 500,
             thin: Optional[int] = 10,
             screen: Optional[bool] = False,
             position: Optional[int] = 0,
             **kwargs):
        """
        Multiple RJ-MCMC steps

        :param n_iter: Number of iterations to perform
        :param n_burn: Number of iterations to burn
        :param thin: Thin chain by this factor - efficient as avoids computing GP predictions
        :param screen: Show detailed state of tree on screen
        :param position: Position of status bar on screen
        """
        if screen:
            stack = Stack()
            pbar = tqdm(total=n_iter, ncols=get_screen_size()[0] - 4,
                        unit=" steps", leave=False)
        else:
            pbar = tqdm(total=n_iter, unit=" steps", position=position)
            pbar.display()

        top_context = stack(0.5) if screen else nullcontext()
        middle_context = stack(0.075) if screen else nullcontext()
        bottom_context = stack(0.075) if screen else nullcontext()

        with top_context as top, middle_context as middle, bottom_context as bottom:

            for iter_ in range(n_iter):

                self.step(**kwargs)

                burn = iter_ < n_burn
                compute = not burn and iter_ % thin == 0

                if compute:
                    self.accumulate()

                pbar.set_description(
                    "burn-in" if burn else "burn-in completed")
                pbar.update()

                if screen:
                    top.write(str(self.tree))
                    middle.write(f"acceptance. {self.acceptance}")
                    bottom.write(str(pbar).strip())

    def evaluate(self, name, log_alpha, systematic=None, loglike=None):
        """
        Evaluate candidate step and record state
        """
        accept = np.log(self.random_state.random()) < log_alpha
        self.acceptance[name].add(accept)

        if accept:
            if systematic is not None:
                self.systematic = systematic
            if loglike is not None:
                self.loglike = loglike

        return accept

    def step(self, n_iter_params=10, min_data_points=1):
        """
        One step, which is a round of trans-dimensional
        and ordinary Markov moves

        :param n_iter_params: Number of parameter iterations per step
        :param min_data_points: Minimum number of data points per leaf
        """
        self.grow_step(min_data_points)
        self.prune_step()
        self.rotate_step()
        for _ in range(n_iter_params):
            self.change_step(min_data_points)
            self.systematic_step()
            self.params_step()

    def change_step(self, min_data_points):
        """
        Change partitions
        """
        if not self.tree.is_leaf():

            parent = draw.choice(self.tree.parents(), self.random_state)
            lower, upper = self.tree.interval(parent)

            original = parent.value
            parent.value = self.change_proposal.rvs(parent.value, self.random_state, lower, upper)
            intervals = self.tree.intervals()

            if min_data_points == 0 or self.model.min_data_points(intervals) >= min_data_points:
                loglike = self.model.loglike(
                    intervals, self.tree.params(), self.systematic)
                log_alpha = loglike - self.loglike
                log_alpha += self.change_proposal.log_pdf_ratio(original, parent.value, lower, upper)
                accept = self.evaluate("change", log_alpha, loglike=loglike)
                if not accept:
                    parent.value = original
            else:
                parent.value = original

    def grow_step(self, min_data_points):
        """
        Grow number of partitions
        """
        n_growable = self.tree.leaf_count
        parent = draw.choice(self.tree.leaves, self.random_state)

        node_params = self.params_prior.rvs(self.random_state)
        value = self.random_state.uniform(*self.tree.interval(parent))
        lhs = draw.bernoulli(self.random_state)

        parent.grow(value, node_params, lhs)

        intervals = self.tree.intervals()

        if min_data_points == 0 or self.model.min_data_points(intervals) >= min_data_points:
            loglike = self.model.loglike(
                intervals, self.tree.params(), self.systematic)

            n_prunable = len(self.tree.prunable())
            log_alpha_grow_ = self.tree_prior.log_pdf_ratio(
                self.tree.get_depth(parent))
            log_alpha_tree = log_alpha_grow_ + np.log(n_growable / n_prunable)
            log_alpha = log_alpha_tree + loglike - self.loglike
            accept = self.evaluate("grow", log_alpha, loglike=loglike)
            if not accept:
                parent.prune(lhs)
        else:
            parent.prune(lhs)

    def prune_step(self):
        """
        Prune number of partitions
        """
        if not self.tree.is_leaf():

            prunable = self.tree.prunable()
            n_prunable = len(prunable)
            parent = draw.choice(prunable, self.random_state)
            value_original = parent.value

            lhs = draw.bernoulli(self.random_state)
            node_params_original = parent.right.node_params if lhs else parent.left.node_params

            parent.prune(lhs)

            loglike = self.model.loglike(
                self.tree.intervals(), self.tree.params(), self.systematic)
            n_growable = self.tree.leaf_count

            log_alpha_grow_ = self.tree_prior.log_pdf_ratio(
                self.tree.get_depth(parent))
            log_alpha_tree = - log_alpha_grow_ + \
                np.log(n_prunable / n_growable)
            log_alpha = log_alpha_tree + loglike - self.loglike

            accept = self.evaluate("prune", log_alpha, loglike=loglike)
            if not accept:
                parent.grow(value_original, node_params_original, lhs)

    def rotate_step(self):
        """
        Rotate the tree
        """
        if not self.tree.is_leaf():

            lhs = draw.bernoulli(self.random_state)
            node = draw.choice(self.tree.parents(), self.random_state)
            depth = self.tree.get_depth(node)
            log_tree_original = self.tree_prior.log_pdf(node, depth)

            if node.is_rotatable(lhs):
                node.rotate(lhs)
                log_alpha = self.tree_prior.log_pdf(
                    node, depth) - log_tree_original
                accept = self.evaluate("rotate", log_alpha)
                if not accept:
                    node.rotate(not lhs)

    def systematic_step(self):
        """
        Markov steps on systematic parameters
        """
        if self.systematic_prior:
            systematic = self.params_proposal.rvs(self.systematic, self.random_state)
            loglike = self.model.loglike(
                self.tree.intervals(), self.tree.params(), systematic)
            log_alpha = (loglike - self.loglike
                         + self.systematic_prior.log_pdf_ratio(self.systematic, systematic)
                         + self.params_proposal.log_pdf_ratio(self.systematic, systematic)
                         )
            self.evaluate("systematic", log_alpha,
                          systematic=systematic, loglike=loglike)

    def params_step(self):
        """
        Markov steps on parameters
        """
        log_alpha = 0.
        node_params_original = []

        for leaf_idx, leaf in enumerate(self.tree.leaves):
            node_params_original.append(leaf.node_params.copy())
            leaf.node_params = self.params_proposal.rvs(
                leaf.node_params, self.random_state)
            log_alpha += self.params_proposal.log_pdf_ratio(
                node_params_original[leaf_idx], leaf.node_params)
            log_alpha += self.params_prior.log_pdf_ratio(
                node_params_original[leaf_idx], leaf.node_params)

        loglike = self.model.loglike(
            self.tree.intervals(), self.tree.params(), self.systematic)
        log_alpha += loglike - self.loglike
        accept = self.evaluate("parameters", log_alpha, loglike=loglike)

        if not accept:
            for leaf_idx, leaf in enumerate(self.tree.leaves):
                leaf.node_params = node_params_original[leaf_idx]
