from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripserplusplus as rpp_py
import numpy as np
import scipy.sparse as sps
import os

mtx = sps.coo_matrix([[0, 5, 0, 0, 0, 0],
                     [5, 0, 0, 7, 0, 12],
                     [0, 0, 0, 0, 0, 0],
                     [0, 7, 0, 0, 22, 0],
                     [0, 0, 0, 22, 0, 0],
                     [0, 12, 0 ,0, 0, 0]])

print(mtx)

args = "--dim 1 --format sparse --threshold 5"
file_name_or_matrix = mtx
d1= rpp_py.run(args, file_name_or_matrix)


args = "--dim 1 --format sparse --threshold 5"
file_name_or_matrix = "sp_test.sparse_matrix"
d2= rpp_py.run(args, file_name_or_matrix)

print(d1)
print(d2)