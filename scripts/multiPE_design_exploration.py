#!/usr/bin/python 
import os
import re
import copy
from config import *
from multiPE_design_generate import Generate_MultiPE_Design, Generate_Host_Code, Generate_Connectivity_Map, Generate_MakeFile

singlePEs_evaluation_file_name = "singlePE_perf_results.log"

with open(singlePEs_evaluation_file_name, 'r') as f:
    # Read the file contents and generate a list with each line
    lines = f.readlines()

singlePE_designs_specs = []
knn_parameters = {}

x = lines[0].split(',')
for item in x:
    knn_parameters[item.strip()] = -1

for idx in range (1, len(lines)):
    singlePE_design = lines[idx]
    y = singlePE_design.split(',')
    this_param = knn_parameters.copy()
    for i in range (len(y)):
        this_param[x[i].strip()] = y[i].strip()
    singlePE_designs_specs.append(this_param)

#enumerate all design choices to determine the max number of PEs base on SLR resource and # banks
max_bw_utilization = 0.0
max_total_num_pe = 0
max_slr_pe = []
max_design_spec = {}
for design in singlePE_designs_specs:
    total_num_PE = 0
    num_PE_SLR = []
    for slr_idx in range(num_SLR):
        num_PE_SLR.append(-1)
        this_slr_num_PE = 99999
        tmp_num_PE = int((float(SLR_resource[slr_idx]['BRAM']) * resource_limit) / float(design['BRAM']))
        this_slr_num_PE = min(this_slr_num_PE, tmp_num_PE)
        tmp_num_PE = int((float(SLR_resource[slr_idx]['DSP']) * resource_limit) / float(design['DSP']))
        this_slr_num_PE = min(this_slr_num_PE, tmp_num_PE)
        tmp_num_PE = int((float(SLR_resource[slr_idx]['FF']) * resource_limit) / float(design['FF']))
        this_slr_num_PE = min(this_slr_num_PE, tmp_num_PE)
        tmp_num_PE = int((float(SLR_resource[slr_idx]['LUT']) * resource_limit) / float(design['LUT']))
        this_slr_num_PE = min(this_slr_num_PE, tmp_num_PE)
        tmp_num_PE = int((float(SLR_resource[slr_idx]['URAM']) * resource_limit) / float(design['URAM']))
        this_slr_num_PE = min(this_slr_num_PE, tmp_num_PE)
        num_PE_SLR[slr_idx] = this_slr_num_PE
        total_num_PE += this_slr_num_PE
    final_num_PE = total_num_PE
    if (total_num_PE*int(design['port_width']) > num_mem_banks*512):
        final_num_PE = num_mem_banks*512/int(design['port_width'])
    if (total_num_PE > final_num_PE):
        offset_PE = total_num_PE - final_num_PE
        for i in range (offset_PE):
            num_PE_SLR[num_PE_SLR.index(max(num_PE_SLR))] -=1
    print 'bw_now:{} bw_before:{}'.format(float(final_num_PE)*float(design['bw_usage']), max_bw_utilization)
    if (float(final_num_PE)*float(design['bw_usage']) > max_bw_utilization):
        max_bw_utilization = float(final_num_PE)*float(design['bw_usage'])
        max_total_num_pe = final_num_PE
        max_slr_pe = copy.deepcopy(num_PE_SLR)
        max_design_spec = design.copy()
    print '{} - {} {} {}'.format(final_num_PE, num_PE_SLR[0], num_PE_SLR[1], num_PE_SLR[2])

print 'bw_usage:{} #PE:{} - {} {} {}'.format(max_bw_utilization, max_total_num_pe, max_slr_pe[0], max_slr_pe[1], max_slr_pe[2])

baseDir = os.getcwd()
genDesignDirName = 'gen_design'
baseDesignDir = os.path.join(baseDir, genDesignDirName)
os.mkdir(baseDesignDir)
os.chdir(baseDesignDir)
os.system("cp -R ../../common .")
os.system("cp ../../utils.mk .")
Generate_MakeFile(num_SLR, kernel_frequency)
designSrcDirName = 'src'
designSrcDir = os.path.join(baseDesignDir, designSrcDirName)
os.mkdir(designSrcDir)
os.chdir(designSrcDir)
Generate_MultiPE_Design(N, D, Dist, K, int(max_design_spec['port_width']), int(max_design_spec['buf_size']), memory_type, max_total_num_pe, max_slr_pe)
pe_bank_location = Generate_Host_Code(memory_type, num_mem_banks, max_total_num_pe, max_slr_pe)
Generate_Connectivity_Map(memory_type, pe_bank_location, max_slr_pe)
