#!/bin/bash

#current directory should be ripser-plusplus
#please source this shell script BEFORE 'source run.sh'

#system requirements: 
#software: the latest version of cmake and cuda (e.g cmake version 3.10.2 and cuda version 9.2.) and gcc version 7.3.0.
#hardware: a GPU with atleast 20GB device DRAM to run ripser++ (tested on a Tesla V100 GPU with 32 GB device memory)

mkdir build

cd build

cmake ..

make

mv ../run.sh run.sh