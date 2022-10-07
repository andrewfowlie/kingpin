"""
Plot helper functions
=====================
"""

import matplotlib.pyplot as plt
import numpy as np


# pylint: disable-next=invalid-name
def pred(x_predict, mean, stdev, ax=None):
    """
    Mean predictions and uncertainty bands
    """
    if ax is None:
        ax = plt.gca()

    ax.fill_between(x_predict, mean + stdev, mean - stdev,
                    alpha=0.95, color="RoyalBlue")
    ax.fill_between(x_predict, mean + 2. * stdev, mean - 2. *
                    stdev, alpha=0.25, color="RoyalBlue")
    ax.plot(x_predict, mean, color="DarkBlue", lw=5)


# pylint: disable-next=invalid-name
def data(x_data, y_data, noise, ax=None):
    """
    Scatter points with error bars for data
    """
    if ax is None:
        ax = plt.gca()

    ax.errorbar(x_data, y_data, yerr=noise, ls='None',
                marker='s', color="Crimson")


# pylint: disable-next=invalid-name
def edges(x_data, edge_counts, ax=None):
    """
    Histogram of edges
    """
    if ax is None:
        ax = plt.gca()

    if np.any(edge_counts):
        ylim = ax.get_ylim()
        center = 0.5 * (x_data[:-1] + x_data[1:])
        width = x_data[1:] - x_data[:-1]
        height = edge_counts / edge_counts.max() * (ylim[1] - ylim[0])
        ax.bar(center, height, width=width,
               bottom=ylim[0], color="grey", alpha=0.5, zorder=0)
