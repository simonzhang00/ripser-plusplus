
from __future__ import print_function
import numpy as np
import ctypes
import math
import sys
import re
import os


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
    print(error_msg, sys.stderr)

    ripserpluspluserror_msg= '''

    Ripser++ Options (for First Argument of run):

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
    '''
    print(ripserpluspluserror_msg, sys.stderr)
    quit()
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

    if len(user_matrix) == 0:
        user_matrix = None

    matrix_formats = ["distance", "lower-distance", "point-cloud", "sparse", "dipha", "binary"]

    # matrix = np.array([])

    # Read from file if not given
    if user_matrix is None:
        if file_format in matrix_formats:

            prog.run_main_filename(len(arguments), arguments, file_name)
            
            return
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
            user_matrix = distance_matrix_user_matrix(user_matrix)

        elif file_format == "lower-distance":
            user_matrix = lower_distance_matrix_user_matrix(user_matrix)

        elif file_format == "point-cloud": # only from file
            point_cloud_user_matrix(user_matrix)
            return

        elif file_format == "dipha": # only from file
            dipha_user_matrix(user_matrix)
            return

        elif file_format == "binary": # only from file
            binary_user_matrix(user_matrix)
            return

        elif file_format == "sparse": # only from file
            binary_user_matrix(user_matrix)
            return

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

        length_of_user_matrix = len(user_matrix)
        user_matrix = (ctypes.c_float * length_of_user_matrix)(*user_matrix)

        prog.run_main(len(arguments), arguments, user_matrix, length_of_user_matrix)

    return


'''
Checks size of user vector
user_vector -- entered user vector
'''
def checkVector(user_vector):

    number_of_points = user_vector.size
    quadratic_sol = (1 + math.sqrt(1 + 8 * number_of_points))/2
    return quadratic_sol*(quadratic_sol-1)/2 == number_of_points

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
    
    # Get lower triangular matrix (not including the 0s)
    indices = np.tril_indices(user_matrix.shape[0], -1)

    # Now convert to vector
    user_matrix = user_matrix[indices]

    # Check size
    check = checkVector(user_matrix)
    if not check:
        printHelpAndExit("User matrix not under the size constraint, number_of_elements = quadtaric_solution * (quadratic_solution-1)/2, where quadratic_solution = (1 + sqrt(1 + 8 * number_of_elements))/2")
        return   

    return user_matrix.flatten()

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
        printHelpAndExit("Vector not under the size constraint, number_of_elements = quadtaric_solution * (quadratic_solution-1)/2, where quadratic_solution = (1 + sqrt(1 + 8 * number_of_elements))/2")
        return

    return user_matrix.flatten()

'''
Runs ripser++ with point_cloud setting using user_matrix
user_matrix -- entered user matrix
'''
def point_cloud_user_matrix(user_matrix):
    printHelpAndExit("point-cloud user matrix is not supported, please load using a file name.")
    return
    

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
file_name -- user file name
'''
def sparse_user_matrix(user_matrix):
    printHelpAndExit("Currently Sparse user matrix is not supported, please load using a file name.")
    return