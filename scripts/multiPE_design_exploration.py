#!/usr/bin/python 
import os
import re
import copy
import math
from config import *
from multiPE_design_generate import Generate_MultiPE_Design
from run_commands import Run_software_emulation
import supporting_code_generation as suppcodegen


def read_singlePE_specs():
    singlePEs_evaluation_file_name = "singlePE_perf_results.log"

    with open(singlePEs_evaluation_file_name, 'r') as f:
        # Read the file contents and generate a list with each line
        lines = f.readlines()

    knn_parameters = {}

    param_names = lines[0].split(',')
    for item in param_names:
        knn_parameters[item.strip()] = -1

    singlePE_synth_values = lines[1].split(',')
    design_specs = knn_parameters.copy()
    for i in range (len(singlePE_synth_values)):
        design_specs[param_names[i].strip()] = singlePE_synth_values[i].strip()

    return design_specs




def create_folders_and_files():
    baseDir = os.getcwd()
    genDesignDirName = 'gen_multiPE_design'
    baseDesignDir = os.path.join(baseDir, genDesignDirName)
    os.mkdir(baseDesignDir)

    ###
    os.chdir(baseDesignDir)
    os.system("cp -R ../../common .")
    os.system("cp ../../utils.mk .")
    suppcodegen.Generate_MakeFile(kernel_frequency, _fpga_part_name = FPGA_part_name)

    designSrcDirName = 'src'
    designSrcDir = os.path.join(baseDesignDir, designSrcDirName)
    os.mkdir(designSrcDir)
    buildDirName = 'build'
    buildDir = os.path.join(baseDesignDir, buildDirName)
    os.mkdir(buildDir)

    ###
    os.chdir(designSrcDir)
    knn_config = Generate_MultiPE_Design(N, D, Dist, K, max_port_width, memory_type, total_num_PE)
    suppcodegen.Generate_Host_Code(total_num_PE, knn_config)
    suppcodegen.Generate_Connectivity_Map(memory_type, total_num_PE, max_port_width)

    ###
    os.chdir(buildDir)
    suppcodegen.Generate_tapa_script(_kernel_freq = kernel_frequency , _fpga_part_name = FPGA_part_name)
    suppcodegen.Generate_hw_build_script(knn_config, _kernel_freq = kernel_frequency)
    suppcodegen.Generate_data_collection_python_script(knn_config)

    ###
    os.chdir(baseDesignDir)
    #Run_software_emulation(DEVICE=FPGA_target_name)
    os.chdir(designSrcDir)






if __name__ == "__main__":

    singlePE_design_specs = read_singlePE_specs()

    #enumerate all design choices to determine the max number of PEs base on SLR resource and # banks
    Total_FPGA_resources = {}

    for resource_type in SLR_resource[0]:
        num_cur_resource = 0
        for i in range(len(SLR_resource)):
            num_cur_resource += SLR_resource[i][resource_type]

        Total_FPGA_resources[resource_type] = num_cur_resource


    ## Initialize to the max possible amount of PEs, given our port width of 512.
    total_num_PE = 32

    if (float(singlePE_design_specs['BRAM']) >= 1):
        tmp_num_PE = int((float(Total_FPGA_resources['BRAM'])) * resource_limit) / float(singlePE_design_specs['BRAM'])
        total_num_PE = min(tmp_num_PE, total_num_PE)
    if (float(singlePE_design_specs['DSP']) >= 1):
        tmp_num_PE = int((float(Total_FPGA_resources['DSP'])) * resource_limit) / float(singlePE_design_specs['DSP'])
        total_num_PE = min(tmp_num_PE, total_num_PE)
    if (float(singlePE_design_specs['FF']) >= 1):
        tmp_num_PE = int((float(Total_FPGA_resources['FF'])) * resource_limit) / float(singlePE_design_specs['FF'])
        total_num_PE = min(tmp_num_PE, total_num_PE)
    if (float(singlePE_design_specs['LUT']) >= 1):
        tmp_num_PE = int((float(Total_FPGA_resources['LUT'])) * resource_limit) / float(singlePE_design_specs['LUT'])
        total_num_PE = min(tmp_num_PE, total_num_PE)
    if (float(singlePE_design_specs['URAM']) >= 1):
        tmp_num_PE = int((float(Total_FPGA_resources['URAM'])) * resource_limit) / float(singlePE_design_specs['URAM'])
        total_num_PE = min(tmp_num_PE, total_num_PE)

    total_num_PE = int(total_num_PE)

    if (total_num_PE > num_mem_banks):
        total_num_PE = int(num_mem_banks)
    print('bw estimate : {}'.format(float(total_num_PE)*float(singlePE_design_specs['bw_usage'])))
    print('Total num PE = {}'.format(total_num_PE), flush=True)


    create_folders_and_files()
