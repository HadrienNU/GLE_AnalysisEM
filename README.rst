.. -*- mode: rst -*-

.. |Travis|_ |Codecov|_

|ReadTheDocs|_ |Black|

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |ReadTheDocs| image:: https://readthedocs.org/projects/gle-analysisem/badge/?version=latest
.. _ReadTheDocs: https://gle-analysisem.readthedocs.io/en/latest/?badge=latest

.. image:: https://zenodo.org/badge/317840103.svg
   :target: https://zenodo.org/badge/latestdoi/317840103

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


GLE_AnalysisEM - A estimator for Generalized Langevin Equation
===============================================================

.. _scikit-learn: https://scikit-learn.org

**GLE_AnalysisEM** is a scikit-learn_ compatible estimator.

Statistical inference of Generalized Langevin Equation using Expectation-Maximization algorithm

To install the module

.. code-block:: sh

	pip3 install --user git+https://github.com/HadrienNU/GLE_AnalysisEM.git

To compile the documentation

.. code-block:: sh

  cd doc/
  make html

And the documentation will be available in

.. code-block:: sh

  doc/_build/html/index.html
