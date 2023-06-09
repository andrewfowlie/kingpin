Welcome to kingpin's documentation!
===================================

.. include :: ../../README.md
   :parser: myst_parser.sphinx_

Theory
^^^^^^

We implement a treed Gaussian process (GP) algorithm :cite:`tgp` in Python. A treed GP automatically partitions the input space and trains a GP in each partition. This allows us to model non-stationary data, heteroscedastic noise and divide and conquer large datasets. The GP hyperparameters and the number and locations of divisions are marginalsed using recursive-jump MCMC.

Example
^^^^^^^

First we import the `kingpin` package

.. code-block:: python

    import kingpin as kp

We use `numpy` to create a dataset

.. code-block:: python

    import numpy as np

    x = np.linspace(0, 10, 101)
    y = 100. * np.ones_like(x)
    y[x > 5.05] = 300.
    y[x > 7.55] = 100.
    noise = np.ones_like(x)

and choose prediction points

.. code-block:: python

    p = np.linspace(x.min(), x.max(), 201)

Now we are ready to make our TGP model. We use the `from_data` constructor. This means that hyperparameter and tree modelling choices are based on peaking at the data

.. code-block:: python

    tgp = kp.TGP.from_data(x, y, noise, p)

Alternatively, all aspects of the model can be chosen by hand. Now we run RJ-MCMC to marginalize the hyperparameters and tree structure

.. code-block:: python

    tgp.walk(n_threads=1, n_iter=1000, n_burn=500)

Finally, we cam take a look at results

.. code-block:: python

    tgp.plot()
    tgp.mean
    tgp.cov

and diagnostics

.. code-block:: python

    tgp.acceptance
    tgp.arviz_summary()

BibTeX
^^^^^^

.. literalinclude:: ../../CITATION.bib
  :language: bibtex

API
^^^

.. toctree::
  api

References
^^^^^^^^^^

.. bibliography::
   :all:


