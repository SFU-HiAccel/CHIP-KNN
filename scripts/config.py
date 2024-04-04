#!/usr/bin/python 

'''---------------------------------------
#       Basic User Configuration
---------------------------------------'''
# KNN Parameters
N = 1024*1024*4
D = 16
Dist = 0 # 0 = Manhattan; 1 = Euclidean
K = 10

USING_HIERMERGE = 1

data_type_to_use = 'float'  # signed fixed, int, or float.

### NOTE: Total number of bits currently must be a power of 2.
###   These parameters are only used if we're using fixed point.
#data_type_int_bits = 12      # Number of bits in the integer part (including sign bit, if signed)
#data_type_fract_bits = 20    # Number of bits in the fractional part

### These values are from running 'platforminfo -p /opt/xilinx/platforms/$(BOARD_NAME)/$(BOARD_NAME).xpfm'.
### For more info, refer to https://www.xilinx.com/support/documentation/boards_and_kits/accelerator-cards/ug1120-alveo-platforms.pdf
SLR_resource_U200 = [{'BRAM':638,  'DSP':2265, 'FF':723372, 'LUT':354831, 'URAM':320}, \
                     {'BRAM':326,  'DSP':1317, 'FF':331711, 'LUT':159854, 'URAM':160}, \
                     {'BRAM':638,  'DSP':2265, 'FF':723353, 'LUT':354962, 'URAM':320}]

SLR_resource_U280 = [{'BRAM':507,  'DSP':2733, 'FF':745576, 'LUT':369145, 'URAM':320}, \
                     {'BRAM':468,  'DSP':2877, 'FF':676164, 'LUT':333217, 'URAM':320}, \
                     {'BRAM':512,  'DSP':2880, 'FF':729059, 'LUT':367546, 'URAM':320}]

resource_limit = 0.6

'''---------------------------------------
       Other Configuration
---------------------------------------'''
SW_EMU_N = 128*1024         # Specifies the number of searchspace points used during the single-PE SW verification.

###########################
###########################
###########################
###########################

# FPGA Platform Specifications
FPGA_target_name = 'xilinx_u280_xdma_201920_3'  # xilinx_u280_xdma_201920_3 or xilinx_u200_xdma_201830_2
FPGA_part_name = 'xcu280-fsvh2892-2L-e' # xcu200-fsgd2104-2-e = U200; xcu280-fsvh2892-2L-e = U280
SLR_resource = SLR_resource_U280
memory_type = 'HBM2'
num_mem_banks = 32
kernel_frequency = 225 #MHz. For U280, this should always be 225.
### Port width (in bits) to off-chip memory
max_port_width = 512   # For the U280, this should always be 512.


if (data_type_to_use == "float"):
    data_type_int_bits = 0
    data_type_fract_bits = 0
