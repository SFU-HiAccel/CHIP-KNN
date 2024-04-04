#!/usr/bin/python 

import os
import re
import json

# Parse_Utilization(_Synthesis_Report_Name)
# Funtion usage: "Parse_Utilization('/Users/aleclu/Desktop/KNN_scripts/csynth.rpt')"
'''
# _Autobridge_Report_Dir: relative path to the Autobridge directory
# Notes: This function gets the HLS usage estimates, which are not perfect. Take the results with a grain of salt.
'''
def Parse_TAPA_Utilization(_TAPA_report_filename):
    resource_types = ['BRAM', 'DSP', 'FF', 'LUT', 'URAM']
    parsed_results = {'BRAM':0, 'DSP':0, 'FF':0, 'LUT':0, 'URAM':0}

    report_file = open(_TAPA_report_filename)
    util_data = json.load(report_file)

    ### Grab the data from TAPA's report.json file
    for r in parsed_results.keys():
        if r == "BRAM":
            parsed_results[r] = util_data["area"]["total"]['BRAM_18K']
        else:
            parsed_results[r] = util_data["area"]["total"][r]

    return parsed_results



def Parse_Autobridge_Utilization(_Autobridge_Report_Dir):

    resource_types = ['BRAM', 'DSP', 'FF', 'LUT', 'URAM']
    parsed_results = {'BRAM':0, 'DSP':0, 'FF':0, 'LUT':0, 'URAM':0}

    for fname in os.listdir(_Autobridge_Report_Dir):
        if "autobridge" in fname:
            autobridge_log = open(_Autobridge_Report_Dir + fname)
            line_list = autobridge_log.readlines()

            for i in range(len(line_list)):
                line = line_list[i]
                if ("The total area" in line):
                    for j in range(len(resource_types)):
                        """
                        ## We assume the autobridge log will output the resource information as follows:

                        The total area of the design:
                          BRAM: 0 / 3504 = 0.0%
                          DSP: 32 / 8496 = 0.4%
                          FF: 28469 / 2331840 = 1.2%
                          LUT: 34281.25 / 1165920 = 2.9%
                          URAM: 0 / 960 = 0.0%
                        """

                        line = line_list[i+j+1]
                        split_line = line.split(":")
                        split_line = split_line[1].split("/")
                        split_line = split_line[0].split()
                        split_line = split_line[0].split(".")
                        resource_usage = int(split_line[0])

                        print("The reported usage for {} is: {}".format(resource_types[j], line))

                        parsed_results[resource_types[j]] = resource_usage
                    break
            break
    
    return parsed_results

