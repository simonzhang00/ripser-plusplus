#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import io
import re
import os
from setuptools import find_packages, setup

DEPENDENCIES = ['numpy']
CURDIR = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(CURDIR, "README.md"), "r", encoding="utf-8") as f:
    README = f.read()

setup(
    name="ripser_plusplus_python",
    version="1.0.0",
    author="Birkan Gokbag - Python Binding, Simon Zhang - Ripser++",
    author_email="birkan.gokbag@gmail.com, szhang31415@gmail.com",
    description="Python binding for Ripser++.",
    long_description=README,
    url="https://github.com/simonzhang00/ripser-plusplus",
    package_dir={'ripser_plusplus_python': 'ripser_plusplus_python'},
    packages=[''],
    include_package_data=True,
    keywords=[],
    scripts=[],
    zip_safe=False,
    install_requires=DEPENDENCIES,
    # license and classifier list:
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    license="License :: OSI Approved :: MIT License",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)