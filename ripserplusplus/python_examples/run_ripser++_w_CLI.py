from __future__ import print_function
import sys
#sys.path.insert(0, '../')#use this to have a reference to ripser_plusplus_python module (module is in parent directory)

# Importing from a previous directory if not installed
import ripserplusplus as rpp_py
import numpy as np
import os


# If the arguments need to be loaded onto the script directly rather than using CLI, can be entered here

file_name = sys.argv[-1:][0]

args = ""
if len(sys.argv) != 1:
    args = ' '.join(sys.argv[1:])
else:
    print("No Arguments Entered for Python Integration w/ Ripser++ CLI, exiting...", sys.stderr)
    sys.exit()

if not os.path.isfile(file_name):
    print("File not found for " + file_name + ", exiting...", sys.stderr)
    sys.exit()

print(rpp_py.run(args, file_name))