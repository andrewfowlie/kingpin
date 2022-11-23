Welcome to kingpin's documentation!
===================================

.. include :: ../../README.md
   :parser: myst_parser.sphinx_

Theory
^^^^^^

We implemen a treed Gaussian process algorithm :cite:`tgp` in Python. A treed GP divides the input space and trains a GP in each region. This allows us to model non-stationary data and divide and conquer large datasets. The GP hyperparameters and the number and locations of divisions are marginalsed using recursive-jump MCMC.

References
^^^^^^^^^^

.. bibliography::
   :all:

BibTeX
^^^^^^

.. literalinclude:: ../../CITATION.bib
  :language: bibtex

API
^^^

.. toctree::
  api
