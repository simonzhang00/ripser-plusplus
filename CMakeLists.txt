cmake_minimum_required(VERSION 3.2)
project(ripser++)

#for Ubuntu users, see: https://github.com/espressomd/espresso/issues/3654 for lowering the gcc version for compatibility with CUDA
find_package(CUDA 7 REQUIRED)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

INCLUDE(CheckCXXSourceCompiles)

include_directories(include)

add_compile_definitions(PRINT_PERSISTENCE_PAIRS)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-G;-Xcompiler;-ggdb;-std=c++11)#manually specify
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-Xcompiler;-ggdb;-std=c++11)
else()
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-gencode;arch=compute_70,code=compute_70;-O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-std=c++11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-Xcompiler;-fopenmp;-Xcompiler;-Ofast;-w;-DNDEBUG;-std=c++11)
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

add_library(phmap STATIC include/phmap_interface/phmap_interface.cpp)

add_executable(ripser ripser.cpp)
cuda_add_executable(ripser++ ripser++.cu)
#target_compile_definitions(ripser++ PUBLIC ASSEMBLE_REDUCTION_SUBMATRIX)
#target_compile_definitions(ripser++ PUBLIC INDICATE_PROGRESS)
#target_compile_definitions(ripser++ PUBLIC CPUONLY_SPARSE_HASHMAP)#GCC version MUST be <=7.3 (tested on version 7.3), Google sparse hashmap no longer supported and thus may not work
#target_compile_definitions(ripser++ PUBLIC CPUONLY_ASSEMBLE_REDUCTION_MATRIX)

target_link_libraries(ripser++ phmap)

# (Optional) script for using this CMakeLists.txt to build libpyripser++.so for Python Bindings in the ripser-plusplus/python/bin folder
#uncomment (one #) per line below to let this cmake build in the ripser-plusplus/python/bin folder
#add_library(phmap_shared SHARED include/phmap_interface/phmap_interface.cpp)
#cuda_add_library(pyripser++ SHARED ripser++.cu)
##target_compile_definitions(pyripser++ PUBLIC ASSEMBLE_REDUCTION_SUBMATRIX)
##target_compile_definitions(pyripser++ PUBLIC INDICATE_PROGRESS)
##target_compile_definitions(pyripser++ PUBLIC CPUONLY_SPARSE_HASHMAP)#GCC version MUST be <=7.3 (tested on version 7.3)
##target_compile_definitions(pyripser++ PUBLIC CPUONLY_ASSEMBLE_REDUCTION_MATRIX)

#target_link_libraries(pyripser++ phmap_shared)

#set_target_properties(pyripser++ PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/bin")

#FUNCTIONAL TESTING:
#https://stackoverflow.com/questions/697560/how-to-copy-directory-from-source-tree-to-binary-tree
add_custom_target(load_testing ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/testing ${CMAKE_BINARY_DIR}/testing
        DEPENDS ${MY_TARGET})

add_custom_target(
        tests
        DEPENDS functional_tests
)

#https://stackoverflow.com/questions/35725750/how-to-execute-a-shell-script-using-cmake-post-build
add_custom_command(OUTPUT functional_tests
        POST_BUILD
        COMMAND /bin/sh ${PROJECT_SOURCE_DIR}/testing/tests.sh
        )

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/testing/run.sh
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
