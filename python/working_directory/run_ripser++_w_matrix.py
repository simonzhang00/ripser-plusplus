from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripser_plusplus_python as rpp_py
import numpy as np
import os

# If the arguments need to be loaded onto the script directly rather than using CLI, can be entered here
args = "--format lower-distance"
# MUST MANUALLY set --format lower-distance if other parameters are being passed
if len(sys.argv) != 1:
    args = ' '.join(sys.argv[1:])
#else:
#    print("No Arguments Entered for Python Integration w/ Ripser ++, exiting...", sys.stderr)
#    sys.exit()

n = 10
print("Running Python Integration w/ Ripser ++", file=sys.stderr)

# Run the program however many times needed
for i in range(10):   
    matrix= np.random.permutation(int((n*(n-1))/2))
    print(matrix)
    rpp_py.run(args, matrix)

'''
DISTANCE MATRIX - USER MATRIX
'''
print("DISTANCE MATRIX - USER MATRIX", sys.stderr)
args = "--dim 1 --format distance"
# Get a square matrix
file_name_or_matrix = np.random.permutation(n * n).reshape((n, n))
# Make it symmetric
file_name_or_matrix = file_name_or_matrix + file_name_or_matrix.T - np.diag(file_name_or_matrix.diagonal())
# Make diagonal be 0
np.fill_diagonal(file_name_or_matrix, 0)

print(file_name_or_matrix)
rpp_py.run(args, file_name_or_matrix)