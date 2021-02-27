from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripserplusplus as rpp_py
import numpy as np
import os

##
## Contains a set of examples to run the python binding for ripser++
##

np.random.seed(1)

############################################
################ FILE NAME ################
##########################################

'''
DISTANCE MATRIX - READ FROM FILE
'''
print("DISTANCE MATRIX - READ FROM FILE", sys.stderr)
args = "--dim 1 --format distance"
file_name_or_matrix = "../examples/celegans.distance_matrix"
rpp_py.run(args, file_name_or_matrix)


'''
LOWER-DISTANCE MATRIX - READ FROM FILE
'''
print("LOWER-DISTANCE MATRIX - READ FROM FILE", sys.stderr)
args = "--dim 1 --format distance --threshold 5"
file_name_or_matrix = "../examples/sphere_3_192.distance_matrix.lower_triangular"
rpp_py.run(args, file_name_or_matrix)


'''
POINT-CLOUD MATRIX - READ FROM FILE
'''
print("POINT-CLOUD MATRIX - READ FROM FILE", sys.stderr)
args = "--dim 1 --format point-cloud --threshold 1.4 --sparse"
file_name_or_matrix = "../examples/o3_4096.point_cloud"
rpp_py.run(args, file_name_or_matrix)


'''
DIPHA MATRIX - READ FROM FILE
'''
#print("DIPHA MATRIX - READ FROM FILE", sys.stderr)
#args = "--dim 1 --format dipha --threshold 5"
#file_name_or_matrix = "../../examples/projective_plane.dipha"
#rpp_py.run(args, file_name_or_matrix)

'''
BINARY MATRIX - READ FROM FILE
'''
#print("BINARY MATRIX - READ FROM FILE, currently no example selected", sys.stderr)
#args = "--dim 1 --format binary --threshold 5"
#file_name_or_matrix = "../../examples/"

'''
SPARSE MATRIX - READ FROM FILE
'''
#print("SPARSE MATRIX - READ FROM FILE", sys.stderr)
#args = "--dim 1 --format sparse"
#file_name_or_matrix = "../../examples/sparse.sparse_matrix"
#rpp_py.run(args, file_name_or_matrix)



############################################
############### USER MATRIX ###############
##########################################

n = 10
'''
DISTANCE MATRIX - USER MATRIX
'''
print("DISTANCE MATRIX - USER MATRIX", sys.stderr)
args = "--dim 1 --format distance --threshold 20"
# Get a square matrix
file_name_or_matrix = np.random.permutation(n * n).reshape((n, n))
# Make it symmetric
file_name_or_matrix = file_name_or_matrix + file_name_or_matrix.T - np.diag(file_name_or_matrix.diagonal())
# Make diagonal be 0
np.fill_diagonal(file_name_or_matrix, 0)

rpp_py.run(args, file_name_or_matrix)

'''
LOWER-DISTANCE MATRIX - USER MATRIX
'''
print("LOWER-DISTANCE MATRIX - USER MATRIX", sys.stderr)
args = "--dim 1 --format lower-distance --threshold 20"
file_name_or_matrix = np.random.permutation(int((n*(n-1))/2))
rpp_py.run(args, file_name_or_matrix)


'''
EXAMPLE FROM THE python/README
'''
rpp_py.run("--format lower-distance",np.transpose(np.array([3,2,1])))
rpp_py.run("--format lower-distance",np.array([3,2,1]))
rpp_py.run("--format distance", np.array([[0,3,2],[3,0,1],[2,1,0]]))

'''
POINT-CLOUD MATRIX - USER MATRIX
'''
#print("POINT-CLOUD MATRIX - USER MATRIX - Currently not available", sys.stderr)
num_pnts= 10
pnt_dimension= 100
rand_point_cloud= np.random.random((num_pnts, pnt_dimension))
rpp_py.run("--format point-cloud", rand_point_cloud)
rpp_py.run("--format point-cloud", np.random.random((3,2)))

rpp_py.run("--format point-cloud", np.array([[1,2,3]]))
rpp_py.run("--format point-cloud", np.array([[1],[2],[3]]))
rpp_py.run("--format point-cloud", np.array([[1,2],[2,3],[3,4]]))

#rpp_py.run("--format distance", np.array([[0]]))

'''
DIPHA MATRIX - USER MATRIX
'''
#print("DIPHA MATRIX - USER MATRIX - Currently not available", sys.stderr)

'''
BINARY MATRIX - USER MATRIX
'''
#print("BINARY MATRIX - USER MATRIX - Currently not available", sys.stderr)

'''
SPARSE MATRIX - USER MATRIX (see sp_tests.py)
'''
#print("SPARSE MATRIX - USER MATRIX - see sp_tests.py", sys.stderr)
