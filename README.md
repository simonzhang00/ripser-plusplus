# Ripser++

[![PyPI license](https://img.shields.io/pypi/l/ripserplusplus.svg)](https://pypi.org/project/ripserplusplus/)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/simonzhang00/44f3f1e65c57d8f4241d34ac83002da9/ripser-plusplus-on-googlecolab.ipynb#scrollTo=nBy0beb9Z1Bi)

Copyright © 2019, 2020, 2021 Simon Zhang, Mengbai Xiao, Hao Wang

Maintainer: Simon Zhang

Contributors:
(by order of introduction to the project)
[Birkan Gokbag](https://github.com/BirkanGokbag), [Ryan DeMilt](https://github.com/ryanpdemilt)

Ripser++ `[3]` is built on top of the Ripser `[1]` software written by Ulrich Bauer and utilizes both GPU and CPU (via separation of parallelisms `[4]`) to accelerate the computation of Vietoris-Rips persistence barcodes.


## Description

Ripser++ utilizes the massive parallelism hidden in the computation of Vietoris-Rips persistence barcodes by taking mathematical and algorithmic oppurtunities we have identified. It can achieve up to 30x speedup over the total execution time of Ripser, up to 2.0x CPU memory efficiency and and up to 1.58x reduction in the amount of memory used on GPU compared to that on CPU for Ripser.

After dimension 0 persistence computation, there are two stages of computation in the original Ripser: filtration construction with clearing followed by matrix reduction. Ripser++ massively parallelizes the filtration construction with clearing stage and extracts the hidden parallelism of finding "apparent pairs" from matrix reduction all on GPU, leaving the computation of submatrix reduction on the remaining nonapparent columns on CPU. By our empirical findings, up to 99.9% of the columns in a cleared coboundary matrix are apparent.


### The Usage of Ripser++ in Other Software Packages
Since the release of Ripser++ in Dec 2019 , the apparent pair search has been reimplemented on CPU directly from our algorithm for:

* An updated [Ripser](https://github.com/Ripser/ripser/tree/b330e04e4995120aa179bedd033bf2df85bd47f3), with the reimplementation written [here](https://github.com/Ripser/ripser/blob/b330e04e4995120aa179bedd033bf2df85bd47f3/ripser.cpp#L540)
* [giotto-ph](https://github.com/giotto-ai/giotto-ph/tree/1d0c628fe5e5c5c4f712bbc1fb9ddc3acc347f9d), a multicore implementation of Ripser, with the reimplementation written [here](https://github.com/giotto-ai/giotto-ph/blob/1d0c628fe5e5c5c4f712bbc1fb9ddc3acc347f9d/gph/src/ripser.h#L766)

## Installation Requirements

Dependencies:

1. a 64 bit Operating System

2. a. Linux

   OR b. Windows

3. CMake >=3.10, (e.g. CMake 3.10.2)

4. CUDA >=10.1, (e.g. CUDA 10.1.243)

5. a. GCC >=7.5, (e.g. GCC 8.4.0 for Linux)

   OR b. MSVC 192x (e.g. MSVC 1928 for Visual Studio 2019 v16.9.2 for Windows)

**Note**: for compilation on Windows, it is best if Cygwin is uninstalled

*Note*: If you turn on the preprocessor directive: `#define CPUONLY_SPARSE_HASHMAP`, then you must lower your GCC version to 7.3.0.
   
Here is a snippet of the table for CUDA/GCC compatibility:
|   CUDA version   | max supported GCC version |
|:----------------:|:-------------------------:|
| 11.1, 11.2, 11.3 |             10            |
|        11        |             9             |
|    10.1, 10.2    |             8             |

And the table for CUDA/MSVC compatibility:
| CUDA version   | Compiler*         | IDE                     |
|----------------|-------------------|-------------------------|
| >=10.1, <=11.3 | MSVC Version 192x | Visual Studio 2019 16.x |


Ripser++ is intended to run on high performance computing systems.

Thus, a GPU with enough device memory is needed to run large datasets. (e.g. Tesla V100 GPU with 32GB device DRAM). If the system's GPU is not compatible, or the system does not have a GPU, error messages will appear.

You do not have to have a super computer, however. On my $900 dollar laptop with a 6GB device memory NVIDIA GPU, I was able to run the sphere_3_192 dataset to dimension 3 computation with a 15x speedup over Ripser.

It is also preferable to have a multicore processor (e.g. >= 28 cores) for high performance, and a large amount of DRAM is required for large datasets. We have tested on a 100 GB DRAM single computing node with 28 cores.

## Installing Python Bindings (preferred)

The purpose of the Python Bindings is to allow users to write their own Python scripts to run Ripser++. The user can write Python preprocessing code on the inputs of Ripser++. This can eliminate file I/O and allow for automated calling of Ripser++.

Contributors:
Ryan DeMilt,
Birkan Gokbag,
Simon Zhang

**Requirements**:

(Requirements from Installation Requirements Section)

Linux, (or Windows), 
CMake >=3.10,
CUDA >=10.1,
GCC >=7.5 (Linux) or Microsoft Visual Studio 2019 (Windows)

Python Requirements:

Python 3.x,
NumPy,
SciPy

(As of January 2020, Python 2.x has been [sunset](https://www.python.org/doc/sunset-python-2/))

## Installation

For the latest release of ripser++:

```
pip3 install git+https://github.com/simonzhang00/ripser-plusplus.git
```

For the version on PyPI:

```
pip3 install ripserplusplus
```

or in the ripser-plusplus/ directory (local installation):

```
git clone https://github.com/simonzhang00/ripser-plusplus.git
pip3 install .
cd ripserplusplus
```

Notice after local installation you need to go to a different directory than ripser-plusplus/ due to path searching in the ```__init__.py``` file.

**Note** Compilation currently can take >=2 minutes on Windows due to a workaround and >=1 minute on Linux so be patient!
**Note** You need all of the software and hardware requirements listed in the installation requirements section.

## The ripserplusplus Python API

ripserplusplus package API:
* Function to Access Ripser++:
    ```
        run(arguments_list, matrix or file_name)
    ```
   * First Argument:
      * arguments_list: Contains the command line options to be entered into Ripser++ as a string. e.g. ```"--format lower-distance --dim 2"```
   * Second Argument: Could be either of the following but not both
      * matrix: Must be a numpy array
         * e.g. ```[3,2,1]``` is a lower-distance matrix of 3 points
         * e.g. ```[[0,3,2],[3,0,1],[2,1,0]]``` is a distance matrix of 3 points
      * or sparse matrix: A scipy coo format matrix
         * e.g. ```mtx = sps.coo_matrix([[0, 5, 0, 0, 0, 0],[5, 0, 0, 7, 0, 12],[0, 0, 0, 0, 0, 0],[0, 7, 0, 0, 22, 0],[0, 0, 0, 22, 0, 0],[0, 12, 0 ,0, 0, 0]])```
      * or file_name: Must be of type string.
         * e.g. ```"../../examples/sphere_3_192.distance_matrix.lower_triangular"```
   * Output: a Python dictionary of numpy arrays of persistence pairs; the dictionary is indexed by the dimension of the array of persistence pairs.
   
Options of Ripser++ for Python bindings:

``` 

Options:

  --help           print this screen
  --format         use the specified file format for the input. Options are:
                     lower-distance (lower triangular distance matrix; default)
                     distance       (full distance matrix)
                     point-cloud    (point cloud in Euclidean space)
                     sparse         (sparse distance matrix in sparse triplet (COO) format)
  --dim <k>        compute persistent homology up to dimension <k>
  --threshold <t>  compute Rips complexes up to diameter <t>
  --sparse         force sparse computation
  --ratio <r>      only show persistence pairs with death/birth ratio > r
```

## How to use Ripser++ with Python Bindings?

Check out the following [gist](https://colab.research.google.com/gist/simonzhang00/44f3f1e65c57d8f4241d34ac83002da9/ripser-plusplus-on-googlecolab.ipynb)
of Ripser++ running on Google Colab.

After having installed the Python bindings successfully (see the Installation section), first checkout the sample code in ripserplusplus/python_examples/ such as examples.py.

To create your own Python script to run Ripser++. Create a Python file (e.g. myExample.py) under ripser-plusplus/python/working_directory/.
At the top of your Python script:

Import the ripserplusplus package to access Ripser++ computing engine:

```
import ripserplusplus as rpp_py
```
Also import numpy, if you want to input a User Matrix:
```
import numpy as np
```
In your Python script, call ```run(arguments_list, matrix or file_name)``` with the following usages:

### Read from File

Python bindings work with file name inputs similar to ripser++ executable. Examples are located under ripser-plusplus/python/working_directory/examples.py.

from the ripser-plusplus/ripserplusplus/ directory:
e.g. ```rpp_py.run("--format point-cloud --sparse --dim 2 --threshold 1.4", "examples/o3_4096.point_cloud")```

### User Matrix Formats

**Note**: default user matrix format is distance in Ripser++. If you know your matrix format is different, then you must use the --format option

#### distance matrix:
* Only supports matrix with the following constraints:
   * Has only 0s at diagonals
   * Symmetric
   * Lower Triangular matrix adhears to the same constraints as lower-distance matrix

e.g. ```rpp_py.run("--format distance", np.array([[0,3,2],[3,0,1],[2,1,0]]))```

runs Ripser++ on a 3 point finite metric space.

#### lower-distance matrix:
* Only supports vectors, as either a row or column vector
* Must be the same size as a square matrix's linearized lower triangular matrix

e.g. ```rpp_py.run("--format lower-distance",np.array([3,2,1]))```

runs Ripser++ on the same data as the distance matrix given above.
#### point-cloud:
* Supports a 2-d numpy array where the number of rows are the number of points embedded in d-dimensional euclidan space and the number of columns is d
* Assumes the Euclidean distance between points

e.g. ```rpp_py.run("--format point-cloud",np.array([[3,2,1],[1,2,3]]))```

runs Ripser++ on a 2 point point cloud in 3 dimensional Euclidean space.

#### sparse (COO):
* Requires SciPy
* Supports a SciPy [coo matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html)

e.g. ```import scipy.sparse as sps; mtx = sps.coo_matrix([[0, 5, 0, 0, 0, 0],[5, 0, 0, 7, 0, 12],[0, 0, 0, 0, 0, 0],[0, 7, 0, 0, 22, 0],[0, 0, 0, 22, 0, 0],[0, 12, 0 ,0, 0, 0]]); rpp_py.run("--format sparse", mtx)```


### Running Python scripts
To run your Python scripts, run, for example, ``` python3 myExample.py``` or ```python3 examples.py``` in the working_directory. This runs Ripser++ through python. A Python dictionary is the output of the run function. Python 2 is no longer supported, please use python3 when running your scripts.


for usage, see the file ripserplusplus/python_examples.py

## How do the Python Bindings Work?

setup.py will build shared object files with CMake: libpyripser++.so and libphmap.so from ripser++.cu. libpyripser++.so is loaded through the ctypes foreign function library of Python. Ripser++ is accessed with the API of one function called ```run(-,-)``` to be called by your own custom Python script.

## Raw Installation From Source

Under a Linux Operating System, type the following commands:

```
git clone https://github.com/simonzhang00/ripser-plusplus.git
cd ripser-plusplus/ripserplusplus
source install_simple.sh
```

The current directory should now be ripser-plusplus/ripserplusplus/build and the executables ripser++, ripser, and the shell script run.sh should show up in the current directory.

To manually build, type the following commands:

```
git clone https://github.com/simonzhang00/ripser-plusplus.git
cd ripser-plusplus/ripserplusplus
mkdir build
cd build
cmake .. && make -j$(nproc)
```

In ripser++.cu there are some preprocessor directives that may be turned on or off. For example, uncomment the line (remove the hash #) : `#target_compile_definitions(ripser++ PUBLIC INDICATE_PROGRESS)` in the CMakeLists.txt file to turn on `INDICATE_PROGRESS`.

The preprocessor directives that can be toggled on are as follows:
- `INDICATE_PROGRESS`: print out the submatrix reduction progress on console; do not use this when redirecting stderr to a file.
- `ASSEMBLE_REDUCTION_SUBMATRIX`: assembles the reduction submatrix on CPU side where the columns in the submatrix correspond to nonapparent columns during submatrix reduction. Oblivious matrix reduction is used by default.
- `CPUONLY_ASSEMBLE_REDUCTION_MATRIX`: assembles the reduction matrix (the sparse V matrix s.t. D*V=R where D is the coboundary matrix) on CPU side for matrix reduction for CPU-only computation if memory allocated for the total possible number of simplices for full Rips computation does not fit into GPU memory.
- `CPUONLY_SPARSE_HASHMAP`: sets the hashmap to Google sparse hashmap during matrix reduction for CPU-only computation if memory allocated for the total possible number of simplices for full Rips computation does not fit into GPU memory. The GCC version must be lowered to <=7.3.0 (tested on 7.3.0) if this option is turned on. (Google sparse hash map is no longer supported and thus may not work)

The only undefined preprocessor directive by default that may improve performance on certain datasets is `ASSEMBLE_REDUCTION_SUBMATRIX`. On certain datasets, especially those where submatrix reduction takes up a large amount of time, the reduction submatrix for submatrix reduction lowers the number of column additions compared to oblivious submatrix reduction at the cost of the overhead of keeping track of the reduction submatrix. However, for most of the datasets we tested on, there is no need to adjust the preprocessor directives for performance.

The following preprocessor directives are defined and may be turned off by manually commenting them out in the code:
- `PRINT_PERSISTENCE_PAIRS`: prints out the persistence pairs to stdout.
- `COUNTING`: prints to stderr the count of different types of columns.
- `PROFILING`: prints to stderr the timing in seconds of different components of computation as well as total GPU memory usage.
- `USE_PHASHMAP`: uses the parallel hashmap library by [Greg Popovitch](https://greg7mdp.github.io/parallel-hashmap/) ; on certain data sets it outperforms the Google sparse hashmap when there are a small number of inserts.

## To Run the Provided Datasets

Please install Ripser++ (see section on raw installation) before trying to run the software.

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

### optional benchmarking
While in the build directory, to compare the performance of Ripser++ with the January 2020 version of [Ripser](https://github.com/Ripser/ripser/tree/286d3696796a707eecd0f71e6377880f60c936da), as done in our paper and run all 6 datasets provided in the examples directory, type:

```
source run.sh
```

**Note**: you will need a very high end GPU to run run.sh effectively. Don't worry about run.sh if you do not have a >20 GB device memory GPU. 

The profiling results should print out.

**Note**: Ripser is very slow (e.g. celegans will take 3-4 minutes to run on Ripser) on these datasets, while Ripser++ will run in seconds on a 32 GB device memory Tesla V100 GPU, so please be patient when the run.sh script runs.

After every command in run.sh has ran, check in your build directory the new directory: run_results. In that directory should be the files (dataset).gpu.barcodes and (dataset).cpu.barcodes for all datasets. (e.g. celegans.gpu.barcodes and celegans.cpu.barcodes) where (dataset).gpu.barcodes are the barcodes of Ripser++ on dataset and (dataset).cpu.barcodes is the profiling of Ripser on (dataset). If you would like to store the profiling results as well, open run.sh and append to the end of each command that runs ripser++: `2> (dataset).gpu.prof` and `2> (dataset).cpu.prof` to the command that runs ripser.

e.g.
```
/usr/bin/time -v ./ripser++ --dim 3 ../examples/celegans.distance_matrix 1> run_results/celegans.gpu.barcodes 2> run_results/celegans.gpu.perf
```

Open the *.gpu.barcodes and *.cpu.barcodes files in the text editor to see the barcodes.

## Running Ripser++ from command line

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

## File Organization

Skeleton directory structure of ripser-plusplus/ripserplusplus/:

```
ripser-plusplus/
   |─ LICENSE        - MIT license 
   |─ CMakeLists.txt - builds the .so files for the python bindings
   |─ MANIFEST.in    - PyPI uploading information
   |─ README.md      - this file
   |─ setup.py       - installs the python bindings
   ripserplusplus/
      |─ examples/ - sample datasets for command line ripser++
      |─ include/  - include files for ripser++.cu
      └─ python_examples/ - Contains examples on how to use the Python bindings, examples are located under under examples.py, and should be used as a working directory (where Python scripts are written) by the user
            |── run_ripser++_w_CLI.py - an example of using a file name to run analysis instead of creating user matrix
            |── run_ripser++_w_matrix.py - an example of creating a user matrix and sending it to Ripser++
            |── examples.py
            ... some other python scripts
            └── your_own_script.py
      |─ testing/                  - testing scripts (use via 'make tests' in the build folder)
      |─ __init__.py               - contains run() 
      |─ CMakeLists.txt            - CMakeLists for building ripser++ command line executable and command line test suite
      |─ install_simple.sh         - use this to build ripser++ for commandline
      |─ ripser++.cu               - the source code
      |─ ripserJan2020             - source code of Ulrich Bauer's original ripser as of Jan 2020 (for reproducibility of our paper's experiments) 
      └─ Ripser_plusplus_Converter - converts python input to ctypes for processing by ripser++
```


## Citing:

```
@inproceedings{zhang2020gpu,
  title={GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes},
  author={Zhang, Simon and Xiao, Mengbai and Wang, Hao},
  booktitle={36th International Symposium on Computational Geometry (SoCG 2020)},
  year={2020},
  organization={Schloss Dagstuhl-Leibniz-Zentrum f{\"u}r Informatik}
}
```

## References:

1. Bauer, Ulrich. "Ripser: efficient computation of Vietoris-Rips persistence barcodes." _arXiv preprint arXiv:1908.02518_ (2019).
2. Otter, Nina, et al. "A roadmap for the computation of persistent homology." _EPJ Data Science_ 6.1 (2017): 17.
3. Zhang, Simon, et al. "GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes." _Proceedings of the Symposium on Computational Geometry_. (SoCG 2020)
4. Zhang, Simon, et al. "HYPHA: a framework based on separation of parallelisms to accelerate persistent homology matrix reduction." _Proceedings of the ACM International Conference on Supercomputing_. ACM, 2019.
