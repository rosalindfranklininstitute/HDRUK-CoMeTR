.. CoMeTR documentation master file, created by
   sphinx-quickstart on Tue Aug 22 14:00:44 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CoMeTR's documentation!
==================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://www.apache.org/licenses/LICENSE-2.0
    :alt: License: Apache 2.0

Overview
--------

CoMeTR (Comprehensive  Metrics for the qualitative and quantitative evaluation and comparison of Tomographic Reconstruction) is a
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





.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

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

Indices and tables
==================

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

We welcome contributions to CoMeTR! If you want to contribute, please contact Dimitrios Bellos via email - Dimitrios.Bellos@rfi.ac.uk or
Laura Shemilt - laura.shemilt@rfi.ac.uk

License
-------

CoMeTR is distributed under the Apache 2.0 license. See the LICENSE file for more details.