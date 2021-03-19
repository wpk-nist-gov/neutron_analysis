================
neutron_analysis
================


.. image:: https://img.shields.io/pypi/v/neutron_analysis.svg
        :target: https://pypi.python.org/pypi/neutron_analysis

.. image:: https://img.shields.io/travis/wpk-nist-gov/neutron_analysis.svg
        :target: https://travis-ci.com/wpk-nist-gov/neutron_analysis

.. image:: https://readthedocs.org/projects/neutron-analysis/badge/?version=latest
        :target: https://neutron-analysis.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




code for analysis of neutron scattering data


* Free software: NIST license
* Documentation: https://neutron-analysis.readthedocs.io.


Features
--------

* Neutron self-shielding calculation
* Photon attenuation calculation


Installation
------------

This project is not yet available via conda or on pypi.  The recommended route is to install most dependencies via conda, then pip install directly from github.  For this, do the following:

Assuming you want to build a new environment, do:

.. code-block:: console

    $ conda create -n {env-name} python=3.8 setuptools pandas xlrd openpyxl lxml periodictable pip

    $ conda activate {env-name}


To install `neutron_analysis` in the active environment do:

.. code-block:: console

   $ pip install git+https://github.com/wpk-nist-gov/neutron_analysis.git@develop


Example usage
-------------

See demo notebook : `demo <notebooks/demo.ipynb>`_


Credits
-------

This package was created with Cookiecutter_ and the `wpk-nist-gov/cookiecutter-pypackage`_ Project template forked from `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`wpk-nist-gov/cookiecutter-pypackage`: https://github.com/wpk-nist-gov/cookiecutter-pypackage
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
