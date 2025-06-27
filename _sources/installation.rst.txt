Installation
============

Requirements
-----------

dominosee requires:

* Python (>=3.9)
* xarray
* dask
* SciPy
* numba
* netCDF4


Installation
-----------

Since dominosee is currently in development, the recommended way to install it is directly from source:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/Hem-W/dominosee-dev.git
    cd dominosee-dev
    
    # Create and activate a conda environment (recommended)
    conda env create -f environment.yml
    conda activate dominosee
    
    # Install the package from source
    pip install -e .


Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development, install with additional development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

Documentation
~~~~~~~~~~~~

To build the documentation:

.. code-block:: bash

    pip install -e ".[docs]"
    cd docs
    make html

Verify Installation
------------------

To verify that dominosee is installed correctly, you can run:

.. code-block:: python

    import dominosee
    print(dominosee.__version__)  # Should print the version number

If you don't see any error messages, you're ready to start using dominosee!
