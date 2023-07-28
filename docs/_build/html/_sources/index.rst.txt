Welcome to CoMeTR's documentation!
==================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://www.apache.org/licenses/LICENSE-2.0
    :alt: License: Apache 2.0

Overview
--------

CoMeTR (Comparison Metrics for Tomographic Reconstruction) is a
gallery of metrics designed to compare volumetric data,
with a focus on tomographic reconstruction techniques.
This documentation provides detailed information about the
CoMeTR Python package and  its usage.

Project Information
-------------------

| **Project Name:** CoMeTR
| **Version:** 0.0.1
| **URL:** `https://github.com/rosalindfranklininstitute/HDRUK-CoMeTR <https://github.com/rosalindfranklininstitute/HDRUK-CoMeTR>`_
| **Author:** Dimitrios Bellos, Netochukwu Onyiaji, Dolapo Adebo
| **Author Email:** dimitrios.bellos@rfi.ac.uk
| **License:** Apache 2.0
| **License File:** LICENSE

Installation
------------

CoMeTR requires Python 3 and the following dependencies:

* numpy
* h5py
* mrcfile
* pyyaml
* beartype
* psutil
* scipy
* scikit-learn

You can install CoMeTR using pip:

.. code-block:: bash

    pip install cometr

Usage
-----

To use CoMeTR, import the relevant modules and functions:

.. code-block:: python

    import cometr

    # Now you can use the CoMeTR functions

.. toctree::
   :maxdepth: 2
   :caption: Contents:

    modules

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Source Code and Issue Tracker
-----------------------------

CoMeTR's source code is hosted on GitHub. You can find the repository and report issues at the following URLs:

| **Source Code:** `https://github.com/rosalindfranklininstitute/HDRUK-CoMeTR <https://github.com/rosalindfranklininstitute/HDRUK-CoMeTR>`_
| **Issue Tracker:** `https://github.com/rosalindfranklininstitute/HDRUK-CoMeTR/issues <https://github.com/rosalindfranklininstitute/HDRUK-CoMeTR/issues>`_

Contributing
------------

We welcome contributions to CoMeTR! If you want to contribute, please contact Dimitrios Bellos via email - Dimitrios.Bellos@rfi.ac.uk

License
-------

CoMeTR is distributed under the Apache 2.0 license. See the LICENSE file for more details.


Command-Line Utilities
----------------------

CoMeTR provides some useful command-line utilities:

* ``cometr.mse``: A utility for calculating the Mean Squared Error (MSE) metric. To use it, run the following command:

.. code-block:: bash

    cometr.mse -f1 <file1> -f2 <file2> -k1 <filekey1> -k2 <filekey2> -f3 <output_text>


    ## Replace <file1> and <file2> with the paths to the input files you want to compare.
    # Replace <filekey1> and <filekey2> with the keys to data in the first and second files, respectively.
    # The MSE result will be stored in <output_text> (default: output.txt).


Testing
-------

To run the tests, you can use pytest:

.. code-block:: bash

    pytest

    # To generate a coverage report:
    pytest --cov=cometr --cov-report = term-missing

Development
-----------

If you are developing CoMeTR, you may want to set up a development environment with the necessary dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

    # This installs the extra dependencies required for development.

Enjoy using CoMeTR!