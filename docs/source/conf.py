import os
import sys

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath('../..'))

# This is to ensure the dominosee module can be found
import dominosee
print(f"Found dominosee module at: {dominosee.__file__}")


# Project information
project = 'DOMINO-SEE'
copyright = '2025, Hui-Min Wang and Xiaogang He (GPL-3.0)'
author = 'Hui-Min Wang and Xiaogang He'
version = '0.0.1'
release = '0.0.1'

# General configuration
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate documentation from docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',        # Support for math equations
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.autosummary',    # Generate summaries automatically
    'sphinx.ext.coverage',       # Check documentation coverage
    'sphinx.ext.todo',           # Support for TODO items
    'sphinx.ext.autosectionlabel', # Auto-generate section labels
    'nbsphinx',                  # Include Jupyter notebooks
    'sphinx_copybutton',         # Add copy button to code blocks
    'jupyter_sphinx',            # Execute code in the documentation
    'sphinx_autodoc_typehints',  # Better handling of type hints in docstrings
    'sphinxcontrib.bibtex',      # Bibliography support
    'sphinx_design',             # Grid layouts and design elements
]

# Add any paths that contain templates
templates_path = ['_templates']

# The suffix of source filenames
source_suffix = ['.rst', '.md']

# The master toctree document
master_doc = 'index'
html_title = "Documentation"

# List of patterns to exclude from source
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'furo'

# Theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo.png",
    "dark_logo": "logo_dark.png",
    "light_css_variables": {
        "color-brand-primary": "#0077B5",        # Professional blue
        "color-brand-content": "#0077B5",
        "color-admonition-background": "#F5F7F9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5CBBFF",        # Lighter blue for dark mode
        "color-brand-content": "#5CBBFF",
        "color-admonition-background": "#242C37",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/PREP-NexT/DOMINO-SEE",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Ensure static files are copied to the root of the build directory
html_extra_path = ['_static']

# Add custom CSS files
html_css_files = [
    'css/custom.css',
]

# Set up intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'dask': ('https://docs.dask.org/en/stable/', None),
}

# Bibliography configuration
bibtex_bibfiles = ['references.bib']

# Include documentation from both class docstring and __init__ docstring
autoclass_content = 'both'

# Default flags used by autodoc directives
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings for NumPy docstrings
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = False
napoleon_use_ivar = True

# Type hints settings (similar to xclim)
autodoc_typehints = 'description'
autodoc_typehints_format = 'fully-qualified'
autodoc_typehints_description_target = 'documented_params'

# Generate autodoc stubs with summaries from code
autosummary_generate = True
autosummary_generate_overwrite = True

# Don't fail on missing module documentation or missing references
nitpicky = False
keep_warnings = False

# Remove the mock import since we want to document the actual code
# autodoc_mock_imports = ['dominosee']

# Skip or mock functions that don't exist yet
def skip_or_mock(app, what, name, obj, skip, options):
    # Handle functions that don't exist yet
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_or_mock)

# Set copybutton to ignore prompts
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Notebook handling - improved configuration for Furo theme
nbsphinx_execute = 'auto'  # Options: 'always', 'never', 'auto' (default)
nbsphinx_allow_errors = True  # Continue building even if notebooks have errors
nbsphinx_timeout = 300  # Give more time for notebook execution

# Notebook styling that works well with Furo
nbsphinx_prompt_width = '0'  # Remove prompt margin
nbsphinx_responsive_width = '100%'  # Make tables responsive

# Add download link for notebooks
nbsphinx_prolog = r"""
{% set docname = 'notebooks/' + env.doc2path(env.docname, base=None).split('/')[-1] %}

.. note:: 

    This page was generated from a Jupyter notebook.
    :download:`Download the notebook <{{ docname }}>`
"""

# Improved styling for notebook outputs in Furo
nbsphinx_codecell_css_classes = ["furo-notebook-cell"]
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Custom CSS for notebooks (merged with existing css files)
html_css_files = [
    'css/custom.css',
    'css/notebook.css',
]
