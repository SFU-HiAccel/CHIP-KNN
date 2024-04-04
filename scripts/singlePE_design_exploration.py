#!/usr/bin/python 
import os
import re
import subprocess
import shutil
from config import *
import bandwidth_interpolation as bw
import utilization_parse as utilparse
from run_commands import Run_tapa_HLS, Run_software_emulation
from singlePE_design_generate import Generate_SinglePE_Design
import supporting_code_generation as suppcodegen


if __name__ == "__main__":
    singlePEs_evaluation_file_name = "singlePE_perf_results.log"
    singlePEs_evaluation_file = []
    column_names = '{:>4}, {:>4}, {:>4}, {:>10}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'.format( \
                    'D', 'Dist', 'K', 'bw_usage', 'BRAM', 'DSP', 'FF', 'LUT', 'URAM')
    singlePEs_evaluation_file.append(column_names + "\n")

    total_num_PE = 1

    baseDir = os.getcwd()
    tmpDesignDirName = 'gen_singlePE_design'
    tmpDesignDirFullPath = os.path.join(baseDir, tmpDesignDirName)

    os.mkdir(tmpDesignDirFullPath)
    os.chdir(tmpDesignDirFullPath)
    os.system("cp -R ../../common .")
    os.system("cp ../../utils.mk .")

    suppcodegen.Generate_MakeFile(kernel_frequency, _fpga_part_name = FPGA_part_name)

    designSrcDirName = 'src'
    designSrcDir = os.path.join(tmpDesignDirFullPath, designSrcDirName)
    os.mkdir(designSrcDir)
    os.chdir(designSrcDir)

    bw_utilization = bw.Interpolate_Bandwidth (memory_type, max_port_width)

    knn_config = Generate_SinglePE_Design(SW_EMU_N, D, Dist, K, max_port_width, memory_type, data_type_to_use, data_type_int_bits, data_type_fract_bits)

    suppcodegen.Generate_Host_Code(total_num_PE, knn_config)
    suppcodegen.Generate_Connectivity_Map(memory_type, total_num_PE, max_port_width)

    os.chdir(tmpDesignDirFullPath)
    Run_software_emulation(DEVICE=FPGA_target_name)

    Run_tapa_HLS(DEVICE=FPGA_target_name)
    #d = utilparse.Parse_TAPA_Utilization('_x.tapa_HLS.xilinx_u200_xdma_201830_2/report.json')
    d = utilparse.Parse_Autobridge_Utilization('_x.tapa_HLS.xilinx_u280_xdma_201920_3/autobridge/')
    design_choice_perf = '{:>4}, {:>4}, {:>4}, {:>10}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'\
                         .format(D, Dist, K, bw_utilization, \
                                 d['BRAM'], d['DSP'], d['FF'], d['LUT'], d['URAM'])
    singlePEs_evaluation_file.append(design_choice_perf + "\n")

    os.chdir(baseDir)
    ### Don't remove the singlePE design dir so the user can examine it, if they want.
    #shutil.rmtree(tmpDesignDirFullPath)

    for line in singlePEs_evaluation_file:
        print("{}".format(line))

    with open(singlePEs_evaluation_file_name, 'w') as f:
            f.seek(0)
            f.writelines(singlePEs_evaluation_file)

    print("SinglePE Exploration Done!")
