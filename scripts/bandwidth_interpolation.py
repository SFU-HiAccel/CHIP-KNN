#!/usr/bin/python 

from config import *

'''
# _memory_type: "DDR4" or "HBM2"
# _port_width: #bit (32 - 512)
'''
def Interpolate_Bandwidth( _memory_type, _port_width):
    _buf_size = 1048576
    port_width_list = [16, 32, 64, 128, 256, 512]     # port width in bits

    buf_size_list = [1024, 2048, 4096, 8192, 16384, \
                      32768, 65536, 131072, 262144, \
                      524288, 1048576] # buffer size in bytes (1KB-1MB)

    # Interpolated bandwidths @ different (port width, buffer size). In GB/s
    DDR4_BW = [[0.44821, 0.51680, 0.55633, 0.57698, 0.58723, 0.59279, 0.59579, 0.59720, 0.59788, 0.59831, 0.59848], \
               [0.89641, 1.03359, 1.11266, 1.15395, 1.17445, 1.18558, 1.19158, 1.19440, 1.19575, 1.19661, 1.19696], \
               [1.49235, 1.78105, 2.05052, 2.22133, 2.30569, 2.34832, 2.37009, 2.38295, 2.38879, 2.39143, 2.39309], \
               [2.25090, 3.07756, 3.59504, 4.15933, 4.47158, 4.62181, 4.70223, 4.74408, 4.76753, 4.77818, 4.78326], \
               [2.87669, 4.53966, 6.10391, 7.24172, 8.31159, 8.86553, 9.20442, 9.40140, 9.48884, 9.53490, 9.55717], \
               [3.30332, 5.66307, 8.71995, 11.7177, 14.4221, 15.8952, 16.8743, 17.4119, 17.7115, 17.8603, 17.9357]]

    HBM2_BW = [[0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0], \
               [0.94898, 1.03389, 1.12187, 1.16194, 1.17894, 1.18997, 1.19479, 1.19746, 1.19867, 1.19937, 1.19966], \
               [1.54376, 1.87831, 2.07197, 2.24275, 2.32440, 2.35766, 2.37981, 2.38977,	2.39492, 2.39737, 2.39876], \
               [2.40349, 3.01884, 3.74049, 4.13425, 4.45530, 4.64984, 4.71468, 4.75987, 4.77932, 4.78962, 4.79469], \
               [3.05019, 4.46640, 5.81290, 6.67501, 7.75598, 8.34681, 9.10847, 9.35273, 9.47944, 9.54084, 9.56996], \
               [3.45862, 5.39038, 7.38211, 8.45044, 9.82558, 11.2194, 12.1991, 12.6875, 12.9825, 13.1056, 13.1761]]

    pw_idx = len(port_width_list)-1
    bs_idx = len(buf_size_list)-1

    for i in range (len(port_width_list)):
        if (port_width_list[i] >= _port_width):
            pw_idx = i
            break
    for i in range (len(buf_size_list)):
        if (buf_size_list[i] >= _buf_size):
            bs_idx = i
            break

    this_BW = []
    if (_memory_type == "DDR4"):
        this_BW = DDR4_BW
    elif (_memory_type == "HBM2"):
        this_BW = HBM2_BW

    if (bs_idx == 0 or bs_idx == len(buf_size_list)-1):
        return this_BW[pw_idx][bs_idx]
    else:
        lo_bs = buf_size_list[bs_idx-1]
        hi_bs = buf_size_list[bs_idx]
        lo_bw = this_BW[pw_idx][bs_idx-1]
        hi_bw = this_BW[pw_idx][bs_idx]
        interpolated_bw = lo_bw + (_buf_size-lo_bs) * (hi_bw-lo_bw) / (hi_bs-lo_bs)
        return interpolated_bw

def Theoretical_Bandwidth (_memory_type, _port_width):
    if (_memory_type == "DDR4"):
        max_bw = 19.2
    elif (_memory_type == "HBM2"):
        max_bw = 14.4

    theoretical_bw = max_bw * (float(_port_width)/max_port_width)
    return theoretical_bw
