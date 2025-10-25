Installation
============

Requirements
------------

DOMINO-SEE requires Python 3.9 or later and the following core dependencies:

* **xarray** (>=0.20.0) - Multi-dimensional labeled arrays
* **dask** - Parallel computing
* **numpy** (>=1.20.0) - Numerical computing
* **scipy** - Scientific computing
* **numba** (>=0.55.0) - JIT compilation for performance
* **netCDF4** - NetCDF file I/O
* **pandas** - Data structures
* **cf-xarray** - CF conventions support
* **bottleneck** - Fast array operations
* **tqdm** - Progress bars

All dependencies are automatically installed when you install DOMINO-SEE.

Installation from Source
-------------------------

DOMINO-SEE is currently in active development. Install directly from the GitHub repository:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/PREP-NexT/DOMINO-SEE.git
    cd DOMINO-SEE
    
    # Create and activate a conda environment (recommended)
    conda env create -f environment.yml
    conda activate dominosee
    
    # Install the package in editable mode
    pip install -e .

The conda environment includes Python 3.11 and all required dependencies.

Optional Dependencies
---------------------

Development Tools
~~~~~~~~~~~~~~~~~

For contributing to DOMINO-SEE, install development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

Documentation
~~~~~~~~~~~~~

To build the documentation locally:

.. code-block:: bash

    pip install -e ".[docs]"
    cd docs
    make html

The built documentation will be available in ``docs/build/html/``.

Verify Installation
-------------------

Verify your installation by importing the package:

.. code-block:: python

    import dominosee
    print(dominosee.__version__)  # Should print: 0.0.1

You can also run the test suite to ensure everything works correctly:

.. code-block:: bash

    pytest tests/

If all tests pass, you're ready to use DOMINO-SEE!
