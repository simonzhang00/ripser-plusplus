# Ripser++

Copyright © 2019, 2020 Simon Zhang, Mengbai Xiao, Hao Wang

Python bindings: Birkan Gokbag

Ripser++ `[3]` is built on top of the Ripser `[1]` software written by Ulrich Bauer and utilizes both GPU and CPU (via separation of parallelisms `[4]`) to accelerate the computation of Vietoris-Rips persistence barcodes.

## Description

Ripser++ utilizes the massive parallelism hidden in the computation of Vietoris-Rips persistence barcodes by taking mathematical and algorithmic oppurtunities we have identified. It can achieve up to 30x speedup over the total execution time of Ripser, up to 2.0x CPU memory efficiency and and up to 1.58x reduction in the amount of memory used on GPU compared to that on CPU for Ripser.

After dimension 0 persistence computation, there are two stages of computation in the original Ripser: filtration construction with clearing followed by matrix reduction. Ripser++ massively parallelizes the filtration construction with clearing stage and extracts the hidden parallelism of finding "apparent pairs" from matrix reduction all on GPU, leaving the computation of submatrix reduction on the remaining nonapparent columns on CPU. By our empirical findings, up to 99.9% of the columns in a cleared coboundary matrix are apparent.

## Installation

Dependencies:

1. Make sure CMake is installed at the correct version (e.g. CMake 3.10.2).

2. Make sure CUDA is installed at the correct version (e.g. CUDA 9.2.88).

3. Make sure GCC is installed at the correct version (e.g. GCC 7.3.0).
    Note: If you turn on the preprocessor directive: `#define CPUONLY_SPARSE_HASHMAP`, then you must lower your GCC version to 7.3.0.
    For CUDA versions higher than 10, GCC must be lowered to atmost 8.
4. A Linux operating system is needed.

Ripser++ is intended to run on high performance computing systems.

Thus, a GPU with enough device memory is needed to run large datasets. (e.g. Tesla V100 GPU with 32GB device DRAM). If the system's GPU is not compatible, or the system does not have a GPU, error messages will appear. 

You do not have to have a super computer, however. On my $900 dollar laptop with a 6GB device memory NVIDIA GPU, I was able to run the sphere_3_192 dataset to dimension 3 computation with a 15x speedup over Ripser. 

It is also preferable to have a multicore processor (e.g. >= 28 cores) for high performance, and a large amount of DRAM is required for large datasets. We have tested on a 100 GB DRAM single computing node with 28 cores.

Under a Linux Operating System, type the following commands:

```
git clone https://github.com/simonzhang00/ripser-plusplus.git
cd ripser-plusplus
source install_simple.sh
```

The current directory should now be ripser-plusplus/build and the executables ripser++, ripser, and the shell script run.sh should show up in the current directory.

To manually build, type the following commands:

```
git clone https://github.com/simonzhang00/ripser-plusplus.git
cd ripser-plusplus
mkdir build
cd build
cmake .. && make -j$(nproc)
```

There are some preprocessor directives that may be turned on or off. For example, uncomment the line (remove the hash #) : `#target_compile_definitions(ripser++ PUBLIC INDICATE_PROGRESS)` in the CMakeLists.txt file to turn on `INDICATE_PROGRESS`.

The preprocessor directives that can be toggled on are as follows:
- `INDICATE_PROGRESS`: print out the submatrix reduction progress on console; do not use this when redirecting stderr to a file.
- `ASSEMBLE_REDUCTION_SUBMATRIX`: assembles the reduction submatrix on CPU side where the columns in the submatrix correspond to nonapparent columns during submatrix reduction. Oblivious matrix reduction is used by default.
- `CPUONLY_ASSEMBLE_REDUCTION_MATRIX`: assembles the reduction matrix (the sparse V matrix s.t. D*V=R where D is the coboundary matrix) on CPU side for matrix reduction for CPU-only computation if memory allocated for the total possible number of simplices for full Rips computation does not fit into GPU memory.
- `CPUONLY_SPARSE_HASHMAP`: sets the hashmap to Google sparse hashmap during matrix reduction for CPU-only computation if memory allocated for the total possible number of simplices for full Rips computation does not fit into GPU memory. The GCC version must be lowered to <=7.3.0 (tested on 7.3.0) if this option is turned on. (Google sparse hash map is no longer supported and thus may not work)

The only undefined preprocessor directive by default that may improve performance on certain datasets is `ASSEMBLE_REDUCTION_SUBMATRIX`. On certain datasets, especially those where submatrix reduction takes up a large amount of time, the reduction submatrix for submatrix reduction lowers the number of column additions compared to oblivious submatrix reduction at the cost of the overhead of keeping track of the reduction submatrix. However, for most of the datasets we tested on, there is no need to adjust the preprocessor directives for performance.

The following preprocessor directives are defined by default and may be turned off by manually commenting them out in the code:
- `PRINT_PERSISTENCE_PAIRS`: prints out the persistence pairs to stdout.
- `COUNTING`: prints to stderr the count of different types of columns.
- `PROFILING`: prints to stderr the timing in seconds of different components of computation as well as total GPU memory usage.
- `USE_PHASHMAP`: uses the parallel hashmap library by [Greg Popovitch](https://greg7mdp.github.io/parallel-hashmap/) ; on certain data sets it outperforms the Google sparse hashmap when there are a small number of inserts.

## To Run the Provided Datasets

Please install Ripser++ (see previous section) before trying to run the software.

Let the current directory be the build directory, then, assuming the above installation procedure worked, type the following command to execute the functional tests to check on installation and GPU compatibility:

```
make tests
```

The functional tests are run with CMake and [shunit2](https://github.com/kward/shunit2). Your system should be compatible with both. All tests are performed with dimension 1 computation.

To see the performance of Ripser++ on a single data set, type the command:

```
./ripser++ --dim 3 ../examples/sphere_3_192.distance_matrix.lower_triangular
```

With a Tesla V100 GPU with 32 GB device memory, this should take just a few seconds (e.g. ~2 to 3 seconds).

While in the build directory, to compare the performance of Ripser++ with the August 2019 version of [Ripser](https://github.com/Ripser/ripser) and run all 6 datasets provided in the examples directory, type:

```
source run.sh
```

The profiling results should print out.

**Note**: Ripser is very slow (e.g. celegans will take 3-4 minutes to run on Ripser) on these datasets, while Ripser++ will run in seconds on a 32 GB device memory Tesla V100 GPU, so please be patient when the run.sh script runs.

After every command in run.sh has ran, check in your build directory the new directory: run_results. In that directory should be the files (dataset).gpu.barcodes and (dataset).cpu.barcodes for all datasets. (e.g. celegans.gpu.barcodes and celegans.cpu.barcodes) where (dataset).gpu.barcodes are the barcodes of Ripser++ on dataset and (dataset).cpu.barcodes is the profiling of Ripser on (dataset). If you would like to store the profiling results as well, open run.sh and append to the end of each command that runs ripser++: `2> (dataset).gpu.prof` and `2> (dataset).cpu.prof` to the command that runs ripser.

e.g.
```
/usr/bin/time -v ./ripser++ --dim 3 ../examples/celegans.distance_matrix 1> run_results/celegans.gpu.barcodes 2> run_results/celegans.gpu.perf
```

Open the *.gpu.barcodes and *.cpu.barcodes files in the text editor to see the barcodes.

In general, to run in the build directory, type:

```
./ripser++ [options] inputfile
```

where inputfile has path relative to the build directory

options:

```
Usage: ripser++ [options] [filename]

Options:

  --help           print this screen
  --format         use the specified file format for the input. Options are:
                     lower-distance (lower triangular distance matrix; default)
                     distance       (full distance matrix)
                     point-cloud    (point cloud in Euclidean space)
                     dipha          (distance matrix in DIPHA file format)
                     sparse         (sparse distance matrix in sparse triplet (COO) format)
                     ripser         (distance matrix in Ripser binary file format)
  --dim <k>        compute persistent homology up to dimension <k>
  --threshold <t>  compute Rips complexes up to diameter <t>
  --sparse         force sparse computation
  --ratio <r>      only show persistence pairs with death/birth ratio > r
```
### Options

The options for Ripser++ are almost the same as those in [Ripser](https://github.com/Ripser/ripser) except for the `--sparse` option.

- `--dim`: specifies the dimension of persistence we compute up to.
- `--threshold`: restricts the diameters of all simplices in the computation, usually paired with the --sparse option.
- `--format`: input formats are the same as those in Ripser. The lower_distance_matrix is the most common input data type format and understood by Ripser++ by default. It is also common to specify a point-cloud in Euclidean space as well.
- `--sparse`: changes the algorithm for computing persistence barcodes and assumes a sparse distance matrix (where many of the distances between points are "infinity"). This can allow for higher dimensional computations and approximations to full Rips persistence computation, so long as the distance matrix is actually sparse. Notice that there is no fail-safe mechanism for the --sparse option when there is not enough GPU device memory to hold the filtration; Thus the program will exit with errors in this case. For the full-Rips case, however, the program can run on CPU-only mode upon discovering there is not enough device memory.
- `--ratio`: only print persistence pairs with death/birth > given ratio

## Datasets:

We provide 6 datasets that are also used in our experiments. For more datasets see [Ulrich Bauer](https://github.com/Ripser/ripser)'s original Ripser repository Github site as well as [Nina Otter](https://github.com/n-otter/PH-roadmap)'s repository on Github for her paper `[2]`. You can generate custom finite metric spaces (distance matrices) or Euclidean space point clouds manually as well.

## Python Bindings

Birkan Gokbag is the author of Python bindings for Ripser++. For information on how to use the Python bindings, consult the README documentation under the ripser-plusplus/python/ directory.

To install the Python bindings from the ripser-plusplus directory, type the command:

```
source install_w_python_bindings.sh
```

This will take you to the ripser-plusplus/python/working_directory/ directory after installation. In this directory you can, for example, write your own Python scripts to preprocess data to form distance matrices and subsequently run Ripser++ on such preprocessed data. In particular, one can apply the approximation algorithm found [here](https://ripser.scikit-tda.org/notebooks/Approximate%20Sparse%20Filtrations.html) to sparsify distance matrices up to some approximation factor on barcode lengths. This can be used when the filtration size becomes unwieldy.

**Note**: you should either run ```source install_w_python_bindings.sh``` from the ripser-plusplus directory every time you start a new session or manually set the PYTHONPATH environment variable for each session to make the Python bindings work (be able to import ripser_plusplus_python). Otherwise, you can add the line: sys.path.insert(0, '../') to any Python script in the working_directory. If you have enough privileges on your system, you can also run ```pip install .``` in the ripser-plusplus/python/ directory.

## Citing:

```
@misc{2003.07989,
Author = {Simon Zhang, Mengbai Xiao, and Hao Wang},
Title = {GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes},
Year = {2020},
Eprint = {arXiv:2003.07989},
}
```

## References:

1. Bauer, Ulrich. "Ripser: efficient computation of Vietoris-Rips persistence barcodes." _arXiv preprint arXiv:1908.02518_ (2019).
2. Otter, Nina, et al. "A roadmap for the computation of persistent homology." _EPJ Data Science_ 6.1 (2017): 17.
3. Zhang, Simon, et al. "GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes." _Proceedings of the Symposium on Computational Geometry_. (SoCG 2020)
4. Zhang, Simon, et al. "HYPHA: a framework based on separation of parallelisms to accelerate persistent homology matrix reduction." _Proceedings of the ACM International Conference on Supercomputing_. ACM, 2019.
