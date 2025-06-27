Contributing
============

We welcome contributions to the dominosee project! This page provides guidelines for contributing to the development of dominosee.

Getting Started
--------------

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine.
3. **Create a branch** for your feature or bugfix.
4. **Make your changes** following the coding guidelines below.
5. **Test your changes** to ensure they work as expected.
6. **Submit a pull request** with a clear description of the changes.

Development Environment
----------------------

To set up your development environment:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/Hem-W/dominosee-dev.git
    cd dominosee-dev
    
    # Create a development environment with conda
    conda env create -f environment.yml
    conda activate dominosee
    
    # Install in development mode
    pip install -e ".[dev]"

Coding Guidelines
----------------

* Follow PEP 8 style guidelines.
* Write docstrings for all functions, classes, and modules following NumPy/SciPy docstring format.
* Include type hints where appropriate.
* Write unit tests for new features.

.. Testing
.. -------

.. Run the tests to make sure your changes don't break existing functionality:

.. .. code-block:: bash

..     # Run tests with pytest
..     pytest

Documentation
-------------

When adding new features, please update the documentation:

1. Add docstrings to your code.
2. Update or create example files if applicable.
3. Build the documentation locally to ensure it renders correctly:

.. code-block:: bash

    # Build documentation
    cd docs
    make html
    
    # View the documentation (opens in browser)
    open build/html/index.html

Pull Request Process
-------------------

1. Update the README.md or documentation with details of your changes.
2. Update the CHANGELOG.md file with details of your changes.
3. Increase the version numbers in any example files and the README.md to the new version that your Pull Request would represent.
4. The PR will be merged once it has been reviewed and approved by a maintainer.
