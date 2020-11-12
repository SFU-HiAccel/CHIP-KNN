#!/usr/bin/python 

import os
import re

# Parse_Utilization(_Synthesis_Report_Name)
# Funtion usage: "Parse_Utilization('/Users/aleclu/Desktop/KNN_scripts/csynth.rpt')"
'''
# _Synthesis_Report_Name: absolute path and HLS synthesis file name
'''
def Parse_Utilization(_Synthesis_Report_Name):
    parsed_results = {'BRAM':-1, 'DSP':-1, 'FF':-1, 'LUT':-1, 'URAM':-1}

    utilization_file_name = _Synthesis_Report_Name
    # Hard-coded parameters
    utilization_est_pattern = 'Utilization Estimates'
    fixed_line_offset = 14

    with open(utilization_file_name, 'r') as f:
        # Read the file contents and generate a list with each line
        lines = f.readlines()

    for idx, line in enumerate(lines):
        match = re.search(utilization_est_pattern, line)
        if match:
            start_line_idx = idx
            break

    x = lines[start_line_idx+fixed_line_offset].split('|')

    parsed_results['BRAM'] = int(x[2])
    parsed_results['DSP'] = int(x[3])
    parsed_results['FF'] = int(x[4])
    parsed_results['LUT'] = int(x[5])
    parsed_results['URAM'] = int(x[6])

    return (parsed_results)