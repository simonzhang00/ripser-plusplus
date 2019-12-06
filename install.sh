#!/bin/bash
#system requirements: 
#software: /usr/bin/time linux command, the lastest version of cmake and cuda (e.g cmake version 3.10.2 and cuda version 9.2.) and gcc version 7.3.0.
#hardware: a GPU with atleast 20GB device DRAM to run ripser++

mkdir build

cd build

cmake ..

make

mv ../run.sh run.sh