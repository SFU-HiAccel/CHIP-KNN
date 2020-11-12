#!/usr/bin/python 

'''---------------------------------------
#       Basic User Configuration
---------------------------------------'''
# KNN Parameters
N = 4194304
D = 2
Dist = 0 # 0 = Manhattan; 1 = Euclidean
K = 10
# FPGA Platform Specifications
FPGA_part_name = 'xcu200-fsgd2104-2-e' # xcu200-fsgd2104-2-e = U200; xcu280-fsvh2892-2L-e = U280
num_SLR = 3
SLR_resource = [{'BRAM':1440, 'DSP':2280, 'FF':788160, 'LUT':394080, 'URAM':320}, \
                {'BRAM':720, 'DSP':1140, 'FF':394080, 'LUT':197040, 'URAM':160}, \
                {'BRAM':1440, 'DSP':2280, 'FF':788160, 'LUT':394080, 'URAM':320}]
memory_type = 'DDR4' # DDR4, HBM2
num_mem_banks = 4
'''---------------------------------------
#       Advanced User Configuration
---------------------------------------'''
singlePE_template_config = [{'port_width':512, 'buf_size':128*1024},\
                            {'port_width':512, 'buf_size':64*1024}] 
                            # {'port_width':512, 'buf_size':64*1024},\
                            # {'port_width':256, 'buf_size':128*1024},\
                            # {'port_width':256, 'buf_size':64*1024}]
resource_limit = 0.7
kernel_frequency = 300 #MHz