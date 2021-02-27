from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripserplusplus as rpp_py
import numpy as np
import scipy.sparse as sps
import os

n = 100
mtx = sps.random(n,n, format = 'coo', density = 0.25)
arr = mtx.todense()
arr = arr + arr.T
np.fill_diagonal(arr,0)

mtx = sps.coo_matrix(arr)
arr2 = sps.coo_matrix.toarray(mtx)

arr = np.tril(arr)

mtx = sps.coo_matrix(arr)

arr = sps.coo_matrix.toarray(mtx)
#print(arr)

'''
SPARSE-DISTANCE MATRIX - READ FROM PYTHON
'''
print("SPARSE-DISTANCE MATRIX - READ FROM PYTHON", sys.stderr)
args = "--dim 1 --format sparse --threshold 5"
file_name_or_matrix = mtx
barcodes_dict = rpp_py.run(args, file_name_or_matrix)

print(barcodes_dict)

