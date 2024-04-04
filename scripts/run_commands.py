#!/usr/bin/python 

import os
import re
from config import *

# Run_HLS_Synthesis(_FPGA_part_name)
# Funtion usage: "Run_HLS_Synthesis('xcu200-fsgd2104-2-e')"
'''
# _FPGA_part_name: FPGA platform part name
'''

def Run_software_emulation(DEVICE):
    print("RUNNING SOFTWARE EMULATION NOW...")
    os.system("make check TARGET=sw_emu DEVICE=" + DEVICE + " 2>&1 | tee SW_EMU_output.log")

def Run_tapa_HLS(DEVICE):
    print("RUNNING SOFTWARE EMULATION NOW...")
    os.system("make tapa_HLS TARGET=tapa_HLS DEVICE=" + DEVICE + " 2>&1 | tee tapa_HLS_output.log")

