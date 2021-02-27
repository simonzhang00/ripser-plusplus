from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripserplusplus as rpp_py
import numpy as np
import os

np.random.seed(1)


'''
LOWER DISTANCE MATRIX - USER MATRIX
'''
# If the arguments need to be loaded onto the script directly rather than using CLI, can be entered here
args = "--format lower-distance"

n = 10
print("Running Python Integration w/ Ripser ++", file=sys.stderr)

# Run the program however many times needed
for i in range(10):
    matrix= np.random.permutation(int((n*(n-1))/2))
    #print(matrix)
    rpp_py.run(args, matrix)
'''
    if(i==0):
        #use this to write out the python data to file to check against commandline ripser++ or ripser for correctness
        np.savetxt('../../build/random_ldm.lower_distance_matrix',matrix)
'''

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

#print(file_name_or_matrix)
rpp_py.run(args, file_name_or_matrix)
'''
#use this to write out the python data to file to check against commandline ripser++ or ripser for correctness
import csv
with open('../../build/random_dm.distance_matrix', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(file_name_or_matrix)
'''

'''
POINT CLOUD MATRIX - USER MATRIX
'''
args = "--dim 2 --format point-cloud"
n = 10
d = 100
matrix = np.random.random((n,d))
'''
#use this to write out the python data to file to check against commandline ripser++ or ripser for correctness
import csv
with open('../../build/random_pc.point_cloud', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(matrix)
'''
rpp_py.run(args, matrix)
