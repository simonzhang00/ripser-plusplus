# Instruction on how to use Python Binding of Ripser++
Author: Birkan Gokbag

Before starting, make sure you read the README for Ripser++ from the ripser-plusplus/ folder as python binding requires Ripser++ requirements to be met.

Requirements:
    Python 2.x / 3.x
    Numpy

## File Organization

**NOTE**: assume the current directory is the project directory ripser-plusplus/ (where ripser++.cu is located).

python/
    bin/ - where all of your executables are located
    ripser_plusplus_python/ - Python Binding package of Ripser++
    working_directory/ - Contains examples on how to use the python binding, examples are located under under examples.py, and should be used as a working directory by the user
        run_ripser++_w_CLI.py - an example of using a file name to run analysis instead of creating user matrix
        run_ripser++_w_matrix.py - an example of creating a user matrix and sending it to ripser++
    README.md - This file
    setup.py - Installs the python package

## Installation

source install_w_python.sh

or:

1) Under ripser-plusplus/python/bin/, run ``` cmake . && make ``` to compile the project and make the executables
2) Set environment variable for the libpyripser++.so with name PYRIPSER_PP_BIN, like:
     ```export PYRIPSER_PP_BIN=$PWD/libpyripser++  ``` under python/bin/
3) Append to PYTHONPATH the location of the directory containing the ripser_plusplus_python package, like:
     ```export PYTHONPATH="${PYTHONPATH}:$PWD/.."```
4) (Optional) Under ripser-plusplus/python/, run ``` pip install . ``` to install the project as a python package. If you are using python3, use pip3.
5) Navigate to ripser-plusplus/python/working_directory/ and check examples.py to see the usage for python integration, and create your own script to run ripser++ with python. The arguments supplied to python integration are identical to the ones supplied to ripser++ executable, it supports user matrices in distance and lower-distance formats.

## Important Notes

* Do not change the directory structure if python package is not installed and environment variable is not set.
* By default, the program assumes the directory for running the program is under ripser-plusplus/python/working_directory/. Make sure to have the relative path to ripser++ and python package to be under the same level.

Example:
python/
    working_directory/
        python_code.py
    ripser_plusplus_python/
        package contents
    bin/
        libpyripser++.so

## How to use Ripser++ with Python Bindings

Import the package to access ripser++ computing engine:

import ripser_plusplus_python as rpp_py

ripser_plusplus_python package:
    User Functions:
        run(arguments_list, matrix or file_name)
        First Argument:
            arguments_list: Contains the arguments to be entered into Ripser++ in text form
        Second Argument: Could be either of the following but not both
            matrix: Must be a numpy array
            file_name: Must be of type string

## Read from File

Python binding works with file name inputs similar to ripser++ executable. Examples are located under ripser-plusplus/working_directory/examples.py

## User Matrix Formats

**Note**: default user matrix format is distance in Ripser++. If you know your matrix format is different, then you must use the --format option

### distance matrix:
* Only supports matrix with the following constraints:
    * Has only 0s at diagonals
    * Symmetric
    * Lower Triangular matrix adhears to the same constraints as lower-distance matrix

### lower-distance matrix:
* Only supports vectors, as either a row or column vector
* Must be the same size as a square matrix's lower triangular matrix

### point-cloud:
* Currently not supported (computing euclidean distances on Python with scipy appears to be much slower than in C++/CUDA), use file name

### sparse (COO):
* Currently not supported, use file name