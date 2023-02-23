#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("lm_classifier", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> List[str]:
    return [
        "tqdm",
        "absl-py",
        "pytorch-lightning",
        "jsonlines",
        "transformers >= 0.7",
        "torch == 1.10.1",
        "torchtext >= 0.9",
        "datasets",
        "scikit-learn",
        "scipy",
        "ipython[notebook]"
    ]


def get_extras_require():
    req = {
        "dev": [
            "sphinxcontrib-bibtex",
            "flake8",
            "flake8-bugbear",
            "yapf",
            "isort",
            "pytest",
            "pytest-cov",
            "mypy",
            "pydocstyle",
            "doc8",
        ],
    }
    return req


setup(
    name="lm_classifier",
    version=get_version(),
    description="The simulated society.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DapangLiu/lm_classifier",
    author="Dapang Research",
    author_email="ruibo.liu.gr@dartmouth.edu",
    license="Apache v2.0",
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="natural language processing, social agent",
    packages=find_packages(
        exclude=["test", "data", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
