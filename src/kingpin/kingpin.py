"""
CLI interface for kingpin
=========================
"""

import click
import numpy as np

from .tgp import TGP


@click.command()
@click.argument('file_name')
@click.option("--seed", default=1, help="Random seed", type=int)
def cli(file_name, seed):
    """
    Read data from file and run RJMCMC
    """
    x_data, y_data = np.loadtxt(file_name, unpack=True)
    tgp = TGP.from_data(x_data, y_data, seed=seed)
    tgp.walk()
    click.echo(tgp.arviz_summary())
    click.echo(tgp.acceptance)
    tgp.show()
