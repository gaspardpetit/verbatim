# Configuration file for the Sphinx documentation builder.
import os
import sys
import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'verbatim'
copyright = f'{datetime.datetime.now().year}, Gaspard Petit'
author = 'Gaspard Petit'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinx_copybutton',
    'myst_parser',
    'sphinx_favicon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = "verbatim"

# -- Favicons ----------------------------------------------------------------
html_favicon = "_static/favicon.ico"
favicons = [
    "favicon-16x16.png",
    "favicon-32x32.png",
    "favicon.ico",
]

# -- MyST Parser settings to work with Markdown files ------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "smartquotes",
    "replacements",
]
myst_heading_anchors = 3
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Autodoc configuration -----------------------------------------------
autoclass_content = 'both'  # Include both class docstring and __init__ docstring
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'  # Show type hints in the description
autodoc_typehints_format = 'short'  # Use short form for type hints

# -- Furo specific settings ---------------------------------------------
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/gaspardpetit/verbatim",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
