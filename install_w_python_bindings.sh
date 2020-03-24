#!/bin/bash

# Change current directory to bin/ under python/
cd python/bin

# Run cmake to compile based on user OS & create executables
cmake ..
make -j$(nproc)

# Set environment variable for python library
export PYRIPSER_PP_BIN=$PWD/libpyripser++.so

echo PYRIPSER_PP_BIN environment variable set to $PYRIPSER_PP_BIN

# append to PYTHONPATH the location of ripser_plusplus_python module
export PYTHONPATH="${PYTHONPATH}:$PWD/.."

echo PYTHONPATH environment variable is now set to $PYTHONPATH

# Install python package for ripser++
# May need sudo permission; also try: pip3 install .
# It is not essential to be able to pip install; everything will still work as long as the ripser_plusplus_python package source code exists.
cd ..
#pip install .

# Go to working_directory at the end
cd working_directory/
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
echo -e ${GREEN}You are currently under the directory:${NC}
echo -e ${BLUE}$PWD${NC}