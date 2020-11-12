#!/usr/bin/python 
import os
import re
import math 
import subprocess
import shutil
from config import *
from bandwidth_interpolation import *
from utilization_parse import *
from hls_run import Run_HLS_Synthesis
from singlePE_design_generate import Generate_SinglePE_Design

singlePEs_evaluation_file_name = "singlePE_perf_results.log"
singlePEs_evaluation_file = []
column_names = '{:>10}, {:>4}, {:>4}, {:>4}, {:>10}, {:>10}, {:>10}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'.format( \
                'N', 'D', 'Dist', 'K', 'port_width', 'buf_size', 'bw_usage', 'BRAM', 'DSP', 'FF', 'LUT', 'URAM')
singlePEs_evaluation_file.append(column_names + "\n")

baseDir = os.getcwd()
tmpDesignDirName = 'tmpDesignDir'
tmpDesignDir = os.path.join(baseDir, tmpDesignDirName)

for choice in singlePE_template_config:
    os.mkdir(tmpDesignDir)
    os.chdir(tmpDesignDir)

    bw_utilization = Interpolate_Bandwidth (memory_type, choice['port_width'], choice['buf_size'])
    Generate_SinglePE_Design(N, D, Dist, K, choice['port_width'], choice['buf_size'], memory_type)
    Run_HLS_Synthesis(FPGA_part_name)
    d = Parse_Utilization('knn.prj/solution0/syn/report/krnl_partialKnn_csynth.rpt')
    design_choice_perf = '{:>10}, {:>4}, {:>4}, {:>4}, {:>10}, {:>10}, {:>10}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'\
                         .format(N, D, Dist, K, choice['port_width'], choice['buf_size'], \
                                 bw_utilization, \
                                 d['BRAM'], d['DSP'], d['FF'], d['LUT'], d['URAM'])
    singlePEs_evaluation_file.append(design_choice_perf + "\n")

    os.chdir(baseDir)
    shutil.rmtree(tmpDesignDir)

with open(singlePEs_evaluation_file_name, 'w') as f:
        f.seek(0)
        f.writelines(singlePEs_evaluation_file)

print "SinglePE Exploration Done!"