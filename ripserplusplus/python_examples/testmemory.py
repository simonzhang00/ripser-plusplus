import ripserplusplus as rpp_py
import subprocess as sp
import os
import numpy as np
import gc

#contributed by: kauii8school
#updated for CPU memory by simonzhang00
#use this file to test the sustainabiltiy of GPU memory over many calls to ripser++

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    #print("free GPU memory (mega): ",memory_free_values)
    print("free GPU memory (MiB):")
    print(memory_free_info[0])
    return memory_free_values
def get_cpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "free --mega"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[0:]
    print("free CPU memory (MiB):")
    print(memory_free_info[0])
    print(memory_free_info[1])
    print(memory_free_info[2])

point_cloud= np.random.rand(1000,3)
#point_cloud= np.random.random((100,100))
for i in range(10000):
    #print(i)
    persistance_dict = rpp_py.run(f'--format point-cloud --dim 1', point_cloud)
    get_gpu_memory()
    get_cpu_memory()
    print("")
    if(i%500==0):
        gc.collect()
