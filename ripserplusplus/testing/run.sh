#!/bin/bash

# this is an experiment measuring the performance improvement of Ripser++ over Ripser

# current directory should be the ripser-plusplus/build folder
# please source this shell script in the ripser-plusplus/build folder AFTER running 'source install_simple.sh' in the ripser-plusplus folder

# datasets to run:
# celegans
# dragon1000
# HIV
# o3_4096
# sphere_3_192
# Vicsek300_300_of_300

# system requirements:
# software: ripser++ and ripser, /usr/bin/time linux command, the latest version of cmake and cuda (e.g cmake version 3.10.2 and cuda version 9.2.) and gcc 7.3.0.
# hardware: a GPU with atleast 20GB device DRAM to run ripser++ (tested on a Tesla V100 GPU with 32 GB device memory)

if [ -e ripser++ -a -e ripser ]
then
    mkdir run_results

    echo RUNNING celegans with Ripser++
    /usr/bin/time -v ./ripser++ --dim 3 ../examples/celegans.distance_matrix 1> run_results/celegans.gpu.barcodes
    echo RUNNING celegans with Ripser
    /usr/bin/time -v ./ripser --dim 3 ../examples/celegans.distance_matrix 1> run_results/celegans.cpu.barcodes

    echo RUNNING dragon1000 with Ripser++
    /usr/bin/time -v ./ripser++ --dim 2 ../examples/dragon1000.distance_matrix 1> run_results/dragon1000.gpu.barcodes
    echo RUNNING dragon1000 with Ripser
    /usr/bin/time -v ./ripser --dim 2 ../examples/dragon1000.distance_matrix 1> run_results/dragon1000.cpu.barcodes

    echo RUNNING HIV with Ripser++
    /usr/bin/time -v ./ripser++ --dim 2 ../examples/HIV.distance_matrix 1> run_results/HIV.gpu.barcodes
    echo RUNNING HIV with Ripser
    /usr/bin/time -v ./ripser --dim 2 ../examples/HIV.distance_matrix 1> run_results/HIV.cpu.barcodes

    echo RUNNING o3_4096 with Ripser++
    /usr/bin/time -v ./ripser++ --threshold 1.4 --format point-cloud --dim 3 --sparse ../examples/o3_4096.point_cloud 1> run_results/o3_4096.gpu.barcodes
    echo RUNNING o3_4096 with Ripser
    /usr/bin/time -v ./ripser --threshold 1.4 --format point-cloud --dim 3 ../examples/o3_4096.point_cloud 1> run_results/o3_4096.cpu.barcodes

    echo RUNNING sphere_3_192 with Ripser++
    /usr/bin/time -v ./ripser++ --dim 3 ../examples/sphere_3_192.distance_matrix.lower_triangular 1> run_results/sphere_3_192.gpu.barcodes
    echo RUNNING sphere_3_192 with Ripser
    /usr/bin/time -v ./ripser --dim 3 ../examples/sphere_3_192.distance_matrix.lower_triangular 1> run_results/sphere_3_192.cpu.barcodes

    echo RUNNING Vicsek300_300_of_300 with Ripser++
    /usr/bin/time -v ./ripser++ --dim 3 ../examples/Vicsek300_300_of_300.distance_matrix 1> run_results/Vicsek300_300_of_300.gpu.barcodes
    echo RUNNING Vicsek300_300_of_300 with Ripser
    /usr/bin/time -v ./ripser --dim 3 ../examples/Vicsek300_300_of_300.distance_matrix 1> run_results/Vicsek300_300_of_300.cpu.barcodes

else
    echo "please run the command 'source install.sh' to build ripser and ripser++ in the build folder before running the command 'source run.sh' in the build folder"
fi