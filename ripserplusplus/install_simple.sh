#!/bin/bash

# Current directory shouild be ripser-plusplus/ripserplusplus
# Please source this shell script BEFORE 'source run.sh'

# System requirements:
# Software: the latest version of cmake and cuda (e.g cmake version 3.10.2 and cuda version 9.2.) and gcc version 7.3.0.
# Hardware: a GPU with atleast 20GB device DRAM to run ripser++ (tested on a Tesla V100 GPU with 32 GB device memory)

mkdir build

cd build

cmake ..

#uncomment the -j$(nproc) optionally for multicore compilation
make #-j$(nproc)

#cp ../testing/run.sh run.sh

GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'
echo -e ${GREEN}You are currently under the directory:${NC}
echo -e ${BLUE}$PWD${NC}
