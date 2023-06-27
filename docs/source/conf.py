# Configuration file for the Sphinx documentation builder.

# -- Project source

import os
import sys

sys.path.insert(0, os.path.abspath('../../src/'))

# -- Project information

project = 'kingpin'
copyright = '2021, Andrew Fowlie'
author = 'Andrew Fowlie'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['../../CITATION.bib']
bibtex_default_style = 'unsrt'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True
autoclass_content = "init"
