#!/usr/bin/python 

'''---------------------------------------
#       Basic User Configuration
---------------------------------------'''
# KNN Parameters
N = 4194304
D = 2
Dist = 1 # 0 = Manhattan; 1 = Euclidean
K = 10
# FPGA Platform Specifications
FPGA_part_name = 'xcu200-fsgd2104-2-e' # xcu200-fsgd2104-2-e = U200; xcu280-fsvh2892-2L-e = U280
num_SLR = 3
SLR_resource = [{'BRAM':1390, 'DSP':2275, 'FF':746000, 'LUT':365000, 'URAM':320}, \
                {'BRAM':752,  'DSP':1317, 'FF':339000, 'LUT':162000, 'URAM':160}, \
                {'BRAM':1390, 'DSP':2275, 'FF':746000, 'LUT':365000, 'URAM':320}]
# SLR_resource_U200 = [{'BRAM':1390, 'DSP':2275, 'FF':746000, 'LUT':365000, 'URAM':320}, \
#                      {'BRAM':752,  'DSP':1317, 'FF':339000, 'LUT':162000, 'URAM':160}, \
#                      {'BRAM':1390, 'DSP':2275, 'FF':746000, 'LUT':365000, 'URAM':320}]
# SLR_resource_U280 = [{'BRAM':980,  'DSP':2733, 'FF':736000, 'LUT':360000, 'URAM':320}, \
#                      {'BRAM':980,  'DSP':2877, 'FF':710000, 'LUT':352000, 'URAM':320}, \
#                      {'BRAM':1020, 'DSP':2800, 'FF':734000, 'LUT':370000, 'URAM':320}]
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