#! /usr/bin/env python
"""Statistical inference of Generalized Langevin Equation using Expectation-Maximization algorithm."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("GLE_analysisEM", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "GLE_AnalysisEM"
DESCRIPTION = "Statistical inference of Generalized Langevin Equation using Expectation-Maximization algorithm."
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "H. Vroylandt"
MAINTAINER_EMAIL = "hadrien.vroylandt@sorbonne-universite.fr"
URL = "https://github.com/HadrienNU/GLE_AnalysisEM"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/HadrienNU/GLE_AnalysisEM"
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "pandas"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {"tests": ["pytest", "pytest-cov"], "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"]}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
