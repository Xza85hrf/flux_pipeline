"""Sphinx configuration for FluxPipeline documentation."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project information
project = "FluxPipeline"
copyright = "2024, Xza85hrf"
author = "Xza85hrf"

# Version info
try:
    from _version import __version__
    version = __version__
    release = __version__
except ImportError:
    version = "0.1.0"
    release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Source file parsers
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Language
language = "en"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type hints
autodoc_typehints = "description"
autodoc_type_aliases = {}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
}

# Todo extension
todo_include_todos = True

# MyST parser options
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "tasklist",
]

# Output
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
