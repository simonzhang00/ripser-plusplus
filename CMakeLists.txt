cmake_minimum_required(VERSION 3.2)
project(pyripser++)

find_package(CUDA 7 REQUIRED)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

INCLUDE(CheckCXXSourceCompiles)

include_directories(ripserplusplus/include)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-G;-Xcompiler;-ggdb;-std=c++11)#manually specify
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-Xcompiler;-ggdb;-std=c++11)
else()
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-gencode;arch=compute_70,code=compute_70;-O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-std=c++11)#-std=c++11;
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-std=c++11)
endif()

include(FindCUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
        "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
        )
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})



FIND_PACKAGE(OpenMP)

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
#include <omp.h>
int main() {
#if (_OPENMP >= 200805 || _MSC_VER >= 1500)
  return 0;
#else
  breaks_on_purpose
#endif
}
" OPENMP_VERSION)

if(OPENMP_VERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("

OpenMP 3.0 not supported by the compiler (${CMAKE_CXX_COMPILER})!
use a C++ compiler with OpenMP 3.0 support (e.g., GCC >=4.4).
To use a different compiler, pass it to cmake in the variable CMAKE_CXX_COMPILER:
 cmake . -DCMAKE_CXX_COMPILER=g++-4.7

")
endif()
set(CMAKE_CXX_STANDARD 11)

#add_library(phmap SHARED ./ripserplusplus/include/phmap_interface/phmap_interface.cpp)
add_library(phmap SHARED ./ripserplusplus/include/phmap_interface/phmap_interface.cpp)

#add_executable(ripser ../ripser.cpp)

cuda_add_library(pyripser++ SHARED ./ripserplusplus/ripser++.cu)
#target_compile_definitions(pyripser++ PUBLIC ASSEMBLE_REDUCTION_SUBMATRIX)
#target_compile_definitions(pyripser++ PUBLIC INDICATE_PROGRESS)
#target_compile_definitions(pyripser++ PUBLIC CPUONLY_SPARSE_HASHMAP)#GCC version MUST be <=7.3 (tested on version 7.3)
#target_compile_definitions(pyripser++ PUBLIC CPUONLY_ASSEMBLE_REDUCTION_MATRIX)

target_link_libraries(pyripser++ phmap)

#cuda_add_executable(ripser++ ../ripser++.cu)
#target_compile_definitions(ripser++ PUBLIC ASSEMBLE_REDUCTION_SUBMATRIX)
#target_compile_definitions(ripser++ PUBLIC INDICATE_PROGRESS)
#target_compile_definitions(ripser++ PUBLIC CPUONLY_SPARSE_HASHMAP)#GCC version MUST be <=7.3 (tested on version 7.3)
#target_compile_definitions(ripser++ PUBLIC CPUONLY_ASSEMBLE_REDUCTION_MATRIX)


#target_link_libraries(ripser++ phmap)
