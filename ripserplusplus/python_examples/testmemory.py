import ripserplusplus as rpp_py
import subprocess as sp
import os
import numpy as np

#contributed by: kauii8school
#use this file to test the sustainabiltiy of GPU memory over many calls to ripser++

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values

point_cloud= np.random.rand(1000,3)
for i in range(10):
    persistance_dict = rpp_py.run(f'--format point-cloud --dim 1', point_cloud)
    get_gpu_memory()
