#!/usr/bin/python 

import os
import re

# Run_HLS_Synthesis(_FPGA_part_name)
# Funtion usage: "Run_HLS_Synthesis('xcu200-fsgd2104-2-e')"
'''
# _FPGA_part_name: FPGA platform part name
'''
def Run_HLS_Synthesis(_FPGA_part_name):
    Gen_HLS_TCL_Script("run_hls.tcl", _FPGA_part_name)
    os.system("vitis_hls run_hls.tcl")

def Gen_HLS_TCL_Script(_TCL_script_name, _FPGA_part_name):
    hls_tcl_file_name = _TCL_script_name
    hls_tcl_file = []
    hls_tcl_file.append("open_project knn.prj" + "\n")
    hls_tcl_file.append("set_top krnl_partialKnn" + "\n")
    hls_tcl_file.append("add_files krnl_partialKnn.cpp" + "\n\n")
    hls_tcl_file.append("open_solution \"solution0\"" + "\n")
    hls_tcl_file.append("set_part {" + _FPGA_part_name +"}" + "\n")
    hls_tcl_file.append("create_clock -period 3.33" + "\n\n")
    hls_tcl_file.append("csynth_design" + "\n")
    hls_tcl_file.append("close_project" + "\n\n")
    hls_tcl_file.append("quit" + "\n")

    with open(hls_tcl_file_name, 'w') as f:
            f.seek(0)
            f.writelines(hls_tcl_file)