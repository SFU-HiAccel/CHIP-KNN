#!/usr/bin/python 

import os
import re
import math 
import sys
from config import *
import knn_code_generation as knngen

'''
# _N: number of search space data points 
# _D: number of query data dimension
# _Dist: type of distance metric: 0=Manhattan 1=Euclidean
# _K: number of nearest neighbors to return 
# _port_width: data port width 
# _memory_type: DDR4 or HBM2
'''
def Generate_SinglePE_Design(_N, _D, _Dist, _K, _port_width, _memory_type, data_type_to_use, data_type_int_bits, data_type_fract_bits):
    total_num_PE = 1
    knn_config = knngen.Generate_Design_Configuration(_N, _D, _Dist, _K, _port_width, _memory_type, \
                                               data_type_to_use, data_type_int_bits, data_type_fract_bits, total_num_PE)

    _Generate_Design_Core(knn_config, total_num_PE, 'krnl_partialKnn') # parallel_sort, num_PE, kernel_name, SLR_name)

    return knn_config


def _Generate_Design_Core(_knn_config, _num_PE, _partialKNN_krnl_name):
    core_krnl_file_name = 'knn.cpp'
    core_krnl_file = []

    knngen.GeneratePartialKNN_Header(core_krnl_file, _knn_config)
    knngen.GeneratePartialKNN_Load(core_krnl_file, _knn_config)
    knngen.GeneratePartialKNN_Compute(core_krnl_file, _knn_config)
    knngen.GeneratePartialKNN_Sort(core_krnl_file, _knn_config)
    knngen.GeneratePartialKNN_HierMerge(core_krnl_file, _knn_config)
    knngen.Generate_singlePE_output_function(core_krnl_file, _knn_config)
    knngen.Generate_KNN_TopLevel(core_krnl_file, _num_PE, _knn_config)

    with open(core_krnl_file_name, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(core_krnl_file)

