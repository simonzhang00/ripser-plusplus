from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripserplusplus as rpp_py
import numpy as np
import os

'''
LOWER-DISTANCE MATRIX - READ FROM FILE
'''
#print("LOWER-DISTANCE MATRIX - READ FROM FILE", sys.stderr)
args = "--dim 1 --format distance --threshold 5"
file_name_or_matrix = "../examples/sphere_3_192.distance_matrix.lower_triangular"
barcodes_dict = rpp_py.run(args, file_name_or_matrix)

print(barcodes_dict)
#./ripser++ --dim 1 --format distance --threshold 5 ../examples/sphere_3_192.distance_matrix.lower_triangular