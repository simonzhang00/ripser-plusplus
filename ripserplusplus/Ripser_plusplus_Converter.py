
from __future__ import print_function
import numpy as np
import scipy.sparse as sps
import ctypes
import math
import sys
import re
import os

class Birth_death_coordinate(ctypes.Structure):
    """
    Replica of datatype for bacrode from cuda
    """
    pass
    _fields_ = [("birth",ctypes.c_float),("death",ctypes.c_float)]
class Set_of_barcodes(ctypes.Structure):
    """
    Replica of datatype for bacrode from cuda
    """
    pass
    _fields_ = [("num_barcodes",ctypes.c_int64),("barcodes",ctypes.POINTER(Birth_death_coordinate))]
class Ripser_plusplus_result(ctypes.Structure):
    """
    Replica of datatype for result from cuda
    """
    pass
    _fields_ = [("num_dimensions",ctypes.c_int64), ("set_of_barcodes",ctypes.POINTER(Set_of_barcodes))]
'''
Prints out the error message and quits the program.
msg -- Custom error message to show the user
'''
def printHelpAndExit(msg):
    error_msg = msg + '''
    How to use Ripser++ with Python Bindings:
    ripser_plusplus_python package:
    User Functions:
        run(arguments_list, matrix or file_name)
                First Argument:
                    arguments_list: Contains the arguments (see Ripser++ options below) to be entered into Ripser++ in text form
                Second Argument: Could be either of the following but not both
                    matrix: Must be a numpy array
                    file_name: Must be of type string
    For more information, please see README.md under ripser_plusplus_python folder.
    '''

    error_msg= error_msg+ '''

    Ripser++ Options (for First Argument of run):

    --help           print this screen
    --format         use the specified file format for the input. Options are:
        lower-distance (lower triangular distance matrix; default)
        distance       (full distance matrix)
        point-cloud    (point cloud in Euclidean space)
        dipha          (distance matrix in DIPHA file format) (not supported)
        sparse         (sparse distance matrix in sparse triplet (COO) format)
        ripser         (distance matrix in Ripser binary file format) (not supported)
    --dim <k>        compute persistent homology up to dimension <k>
    --threshold <t>  compute Rips complexes up to diameter <t>
    --sparse         force sparse computation
    --ratio <r>      only show persistence pairs with death/birth ratio > r
    '''
    #print(error_msg, sys.stderr)
    raise Exception(error_msg)
'''
Searches the path and all its children for file named name
'''
def find(name, path):#stackoverflow.com/questions/1724693/find-a-file-in-python
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

'''
Runs ripser++ using the user arguments.
prog -- loaded in program
arguments -- user arguments
file_name -- user file name, empty if not specified
file_format -- file format for user matrix or file name, default: distance
user_matrix -- Entered user matrix, either a vector or a matrix with at least 2 rows.
'''
def Ripser_plusplus_Converter(prog, arguments, file_name, file_format, user_matrix = None):

    if isinstance(user_matrix,np.ndarray) and len(user_matrix) == 0:
        user_matrix = None
    elif isinstance(user_matrix, sps.coo_matrix) and sps.coo_matrix.getnnz(user_matrix) == 0:
        user_matrix = None
 

    matrix_formats = ["distance", "lower-distance", "point-cloud", "sparse", "dipha", "binary"]

    # Read from file if not given
    if user_matrix is None:
        if file_format in matrix_formats:
            

            prog.run_main_filename.restype = Ripser_plusplus_result
            res = prog.run_main_filename(len(arguments), arguments, file_name)

            barcodes_dict = {}

            for dim in range(res.num_dimensions):
                barcodes_dict[dim] = np.array([np.array(res.set_of_barcodes[dim].barcodes[coord]) for coord in range(res.set_of_barcodes[dim].num_barcodes)])
            return barcodes_dict
            
        else:

            printHelpAndExit("Unknown file format. Please use one of the following\n" +
                                "    distance\n" +
                                "    lower-distance\n" +
                                "    point-cloud\n" +
                                "    dipha\n" +
                                "    binary\n" +
                                "    sparse\n")
            return

    else:

        if file_format == "distance":
            num_entries, num_rows, num_columns, user_matrix = distance_matrix_user_matrix(user_matrix)

        elif file_format == "lower-distance":
            num_entries, num_rows, num_columns, user_matrix = lower_distance_matrix_user_matrix(user_matrix)

        elif file_format == "point-cloud":
            num_entries, num_rows, num_columns, user_matrix = point_cloud_user_matrix(user_matrix)

        elif file_format == "dipha": # only from file
            dipha_user_matrix(user_matrix)
            return

        elif file_format == "binary": # only from file
            binary_user_matrix(user_matrix)
            return

        elif file_format == "sparse": # only from file
            num_entries, num_rows, num_columns, user_matrix = sparse_user_matrix(user_matrix)

        else:
            printHelpAndExit("Unknown file format. Please use one of the following\n" +
                                "    distance\n" +
                                "    lower-distance\n" +
                                "    point-cloud\n" +
                                "    dipha\n" +
                                "    binary\n" +
                                "    sparse\n")
            return

        if user_matrix is None:
            printHelpAndExit("User Matrix was not created")
            return

        user_matrix = (ctypes.c_float * num_entries)(*user_matrix)

        prog.run_main.restype = Ripser_plusplus_result

        barcodes_dict = {}

        res = prog.run_main(len(arguments), arguments, user_matrix, num_entries, num_rows, num_columns)

        for dim in range(res.num_dimensions):
            barcodes_dict[dim] = np.array([np.array(res.set_of_barcodes[dim].barcodes[coord]) for coord in range(res.set_of_barcodes[dim].num_barcodes)])
        return barcodes_dict
        

    return


'''
Checks size of user vector
user_vector -- entered user vector
'''
def checkVector(user_vector):

    number_of_entries = user_vector.size
    quadratic_sol = (1 + math.sqrt(1 + 8 * number_of_entries))/2
    return quadratic_sol*(quadratic_sol-1)/2 == number_of_entries

# User Matrices

'''
Runs ripser++ with distance setting using user_matrix
user_matrix -- entered user matrix
'''
def distance_matrix_user_matrix(user_matrix):

    # First check if symmetric
    if not np.allclose(user_matrix, user_matrix.T):
        printHelpAndExit("Entered user matrix needs to be symmetric")
        return

    # Check if diagonals are all 0
    if not (np.diagonal(user_matrix) == [0]).all():
        printHelpAndExit("All diagonals need to be 0 for the distance matrix.")
        return

    if len(user_matrix.shape)!=2:
        printHelpAndExit("Distance matrix must be an actual 2-d matrix")
        return

    # Get lower triangular matrix (not including the 0s)
    indices = np.tril_indices(user_matrix.shape[0], -1)

    num_rows, num_columns = user_matrix.shape
    if num_rows != num_columns:
        printHelpAndExit("Distance matrix must be square")
        return
    # Now convert to vector
    user_matrix = user_matrix[indices]

    # Check size
    check = checkVector(user_matrix)
    if not check:
        printHelpAndExit("User matrix not under the size constraint, number_of_elements = quadratic_solution * (quadratic_solution-1)/2, where quadratic_solution = (1 + sqrt(1 + 8 * number_of_elements))/2")
        return

    num_entries= len(user_matrix)

    return num_entries, num_rows, num_columns, user_matrix

'''
Runs ripser++ with lower-distance setting using user_matrix
user_matrix -- entered user matrix
'''
def lower_distance_matrix_user_matrix(user_matrix):

    user_matrix_dimensions = user_matrix.shape
    # Check whether it is a row/column vector
    #if not(len(user_matrix_dimensions) == 1 or user_matrix_dimensions[0] == 1 or user_matrix_dimensions[1] == 1):
    if(len(user_matrix_dimensions)!=1):
        printHelpAndExit("Lower Distance Matrix only supports a vector\n")
        return
    user_matrix = user_matrix.reshape((1,user_matrix.size))

    # Check size
    check = checkVector(user_matrix)
    if not check:
        printHelpAndExit("Vector not under the size constraint, number_of_elements = quadratic_solution * (quadratic_solution-1)/2, where quadratic_solution = (1 + sqrt(1 + 8 * number_of_elements))/2")
        return
    user_matrix = user_matrix.ravel()
    num_entries = len(user_matrix)
    num_rows = ctypes.c_int(int((1 + math.sqrt(1 + 8 * num_entries))/2))
    num_columns = num_rows
    return num_entries, num_rows, num_columns, user_matrix

'''
Runs ripser++ with point_cloud setting using user_matrix
user_matrix -- entered user matrix
'''
def point_cloud_user_matrix(user_matrix):
    #printHelpAndExit("Currently point-cloud user matrix is not supported, please load using a file name.")
    num_rows, num_columns= user_matrix.shape
    user_matrix= user_matrix.ravel()
    num_entries= len(user_matrix)
    if(num_entries==0):
        printHelpAndExit("point-cloud needs at least one point")
        return
    return num_entries, num_rows, num_columns, user_matrix
    

'''
Runs ripser++ with dipha setting using user_matrix
user_matrix -- entered user matrix
'''
def dipha_user_matrix(user_matrix):
    printHelpAndExit("Dipha user matrix is not supported, please load using a file name.")
    return

'''
Runs ripser++ with binary setting using user_matrix
user_matrix -- entered user matrix
'''
def binary_user_matrix(user_matrix):
    printHelpAndExit("Binary user matrix is not supported, please load using a file name.")
    return


'''
Runs ripser++ with sparse setting using user_matrix
user_matrix -- entered user matrix
'''
def sparse_user_matrix(user_matrix):
    num_rows, num_columns = user_matrix.shape
    user_matrix = np.array([value for pair in zip(user_matrix.row,user_matrix.col,user_matrix.data) for value in pair])
    num_entries = len(user_matrix)

    if(num_entries == 0):
        printHelpAndExit("Sparse matrix needs at least one entry")
        return
    
    return num_entries, num_rows, num_columns, user_matrix