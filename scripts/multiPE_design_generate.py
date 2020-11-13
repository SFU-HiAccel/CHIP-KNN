#!/usr/bin/python 

import os
import re
import math 
from bandwidth_interpolation import *

# Generate_MultiPE_Design(_N, _D, _Dist, _K, _port_width, _buf_size, _total_num_pe, _slr_num_pe)
# Funtion usage: "Generate_MultiPE_Design(4194304, 2, 0, 10, 512, 131072, 8, [3,2,3])"
'''
# _N: number of search space data points 
# _D: number of query data dimension
# _Dist: type of distance metric: 0=Manhattan 1=Euclidean
# _K: number of nearest neighbors to return 
# _port_width: data port width 
# _buf_size: on-chip BRAM buffer size 
# _memory_type: DDR4 or HBM2
# _total_num_pe: total number of PEs across all SLRs
# _slr_num_pe: number of PEs for each SLR 
'''
def Generate_MultiPE_Design(_N, _D, _Dist, _K, _port_width, _buf_size, _memory_type, _total_num_pe, _slr_num_pe):
    parallel_sort = Generate_Design_Configuration(_N, _D, _Dist, _K, _port_width, _buf_size, _memory_type, _total_num_pe, _slr_num_pe)
    for slr_idx in range (len(_slr_num_pe)):
        Generate_PartialKNN_Design(parallel_sort, _slr_num_pe[slr_idx], 'krnl_partialKnn_SLR'+str(slr_idx), 'SLR'+str(slr_idx)) # parallel_sort, num_PE, kernel_name, SLR_name)
    Generate_GlobalSort_Design(_slr_num_pe)

def Generate_Design_Configuration(_N, _D, _Dist, _K, _port_width, _buf_size, _memory_type, _total_num_pe, _slr_num_pe):
    config_file = 'krnl_config.h'

    INPUT_DIM = _D
    TOP = _K
    NUM_SP_PTS = _N
    DISTANCE_METRIC = _Dist

    NUM_KERNEL = _total_num_pe

    NUM_USED_SLR = 0
    for i in range (len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            NUM_USED_SLR += 1
    
    DWIDTH = _port_width
    BUFFER_SIZE_DEFAULT = _buf_size * 8 #convert to # of bit
    MAX_FLT = "3.402823e+38f"

    DIS_CALCULATION_DEPTH = 125
    SORT_II = 3

    bw_utilization = Interpolate_Bandwidth (_memory_type, _port_width, _buf_size)
    theorectical_bw_utilization = Theoretical_Bandwidth (_memory_type, _port_width)
    factor = theorectical_bw_utilization/bw_utilization
    BW_FACTOR = factor #default=1.2

    new_file = []
    baseDirName = os.getcwd()
    new_file.append("// PWD:" + baseDirName + "\n\n")

    new_file.append("#include \"ap_int.h\" \n")
    new_file.append("#include \"ap_axi_sdata.h\" \n")
    new_file.append("#include \"hls_stream.h\" \n")
    new_file.append("#include <inttypes.h> \n")
    new_file.append("#include <math.h> \n")
    new_file.append("#include <stdlib.h> \n")

    new_file.append("\n")

    new_file.append("const int DWIDTH = " + str(DWIDTH) + "; \n")
    new_file.append("#define INTERFACE_WIDTH ap_uint<DWIDTH> \n")
    new_file.append("const int SWIDTH = 32; \n")
    new_file.append("typedef ap_axiu<SWIDTH, 0, 0, 0> pkt; \n")
    new_file.append("#define STREAM_WIDTH ap_uint<SWIDTH> \n")

    new_file.append("\n")

    new_file.append("#define INPUT_DIM " + "(" + str(INPUT_DIM) + ")" + "\n")
    new_file.append("#define TOP " + "(" + str(TOP) + ")" + "\n")
    new_file.append("#define NUM_SP_PTS " + "(" + str(NUM_SP_PTS) + ")" + "\n")

    NUM_SP_PTS_PER_KRNL = math.ceil(NUM_SP_PTS / NUM_KERNEL)
    NUM_OF_TILES = math.floor(NUM_SP_PTS_PER_KRNL * INPUT_DIM * 32 / BUFFER_SIZE_DEFAULT)
    BUFFER_SIZE_PADDED = math.ceil(NUM_SP_PTS_PER_KRNL * INPUT_DIM * 32 / NUM_OF_TILES / 1024 / 8) * 1024 * 8 
    NUM_SP_PTS_PER_KRNL_PADDED = BUFFER_SIZE_PADDED * NUM_OF_TILES / INPUT_DIM / 32
    NUM_SP_PTS_PADDED = NUM_SP_PTS_PER_KRNL_PADDED * NUM_KERNEL

    new_file.append("#define NUM_SP_PTS_PADDED " + "(" + str(int(NUM_SP_PTS_PADDED)) + ")" + "\n")
    new_file.append("#define DISTANCE_METRIC " + "(" + str(DISTANCE_METRIC) + ")" + "\n")
    new_file.append("#define NUM_KERNEL " + "(" + str(NUM_KERNEL) + ")" + "\n")
    new_file.append("#define NUM_USED_SLR " + "(" + str(NUM_USED_SLR) + ")" + "\n")
    new_file.append("#define MAX_FLT " + "(" + MAX_FLT + ")" + "\n")

    new_file.append("\n")

    SP_LEN = BUFFER_SIZE_PADDED / DWIDTH;
    DIS_LEN = BUFFER_SIZE_PADDED / (32 * INPUT_DIM);
    new_file.append("#define SP_LEN " + "(" + str(int(SP_LEN)) + ")" + "\n")
    new_file.append("#define DIS_LEN " + "(" + str(int(DIS_LEN)) + ")" + "\n")
    new_file.append("#define NUM_OF_TILES " + "(" + str(int(NUM_OF_TILES)) + ")" + "\n")

    NUM_FEATURES_PER_READ = DWIDTH / 32
    new_file.append("const int NUM_FEATURES_PER_READ = " + str(int(NUM_FEATURES_PER_READ)) + ";\n")
    QUERY_FEATURE_RESERVE = 128
    new_file.append("const int QUERY_FEATURE_RESERVE = " + str(int(QUERY_FEATURE_RESERVE)) + ";\n")
    QUERY_DATA_RESERVE = QUERY_FEATURE_RESERVE / NUM_FEATURES_PER_READ
    new_file.append("#define QUERY_DATA_RESERVE " + "(" + str(int(QUERY_DATA_RESERVE)) + ")" + "\n")

    if (DWIDTH / (32 * INPUT_DIM) >= 1):
        PARALLEL_SORT = 1
        FACTOR_W = DWIDTH / (32 * INPUT_DIM)
        PARALLEL_SORT_FACTOR = FACTOR_W * SORT_II
        PARALLEL_SORT_SIZE = math.ceil(math.ceil(DIS_LEN/PARALLEL_SORT_FACTOR)/FACTOR_W) * FACTOR_W
        START_OF_PADDING = DIS_LEN % PARALLEL_SORT_SIZE
        new_file.append("#define PARALLEL_SORT " + "(" + str(PARALLEL_SORT) + ")" + "\n")
        new_file.append("const int FACTOR_W = " + str(int(FACTOR_W)) + ";\n")
        new_file.append("#define PARALLEL_SORT_FACTOR " + "(" + str(int(PARALLEL_SORT_FACTOR)) + ")" + "\n")
        new_file.append("#define PARALLEL_SORT_SIZE " + "(" + str(int(PARALLEL_SORT_SIZE)) + ")" + "\n")
        new_file.append("#define START_OF_PADDING " + "(" + str(int(START_OF_PADDING)) + ")" + "\n")
    else:
        PARALLEL_SORT = 0
        FACTOR_W = (32 * INPUT_DIM) / DWIDTH
        DIS_CALCULATION_II = math.floor((SP_LEN * BW_FACTOR - DIS_CALCULATION_DEPTH * FACTOR_W) / DIS_LEN)
        new_file.append("#define PARALLEL_SORT " + "(" + str(PARALLEL_SORT) + ")" + "\n")
        new_file.append("const int FACTOR_W = " + str(int(FACTOR_W)) + ";\n")    
        new_file.append("const int DIS_CALCULATION_II = " + str(int(DIS_CALCULATION_II)) + ";\n")

    new_file.append("\n")
    NUM_ITERATIONS = 5000
    new_file.append("const int NUM_ITERATIONS = " + str(int(NUM_ITERATIONS)) + ";\n")
        
    with open(config_file, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(new_file)

    return PARALLEL_SORT

def Generate_PartialKNN_Design(_parallel_sort, _num_PE, _krnl_top_name, _SLR_name):
    if (_num_PE < 1):
        return
    core_krnl_file_name = _krnl_top_name + '.cpp'
    core_krnl_file = []

    GenerateHeader(core_krnl_file, _num_PE)
    GenerateLoad(core_krnl_file, _num_PE, _SLR_name)
    GenerateCompute(core_krnl_file, _parallel_sort, _SLR_name)
    GenerateSort(core_krnl_file, _parallel_sort, _SLR_name)
    if (_num_PE > 1):
        GenerateGlobalMerge(core_krnl_file, _SLR_name)
    GenerateTopLevel(core_krnl_file, _parallel_sort, _num_PE, _krnl_top_name, _SLR_name)

    with open(core_krnl_file_name, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(core_krnl_file)

def GenerateHeader(_core_krnl_file, _num_PE):
    _core_krnl_file.append('#include "krnl_config.h"' + '\n')
    _core_krnl_file.append('#define NUM_PART (' + str(_num_PE) + ')' + '\n')
    _core_krnl_file.append('extern "C" {'  + '\n\n')

def GenerateLoad(_core_krnl_file, _num_PE, _SLR_name):
    for idx in range (_num_PE):
        _core_krnl_file.append("void load_" + _SLR_name + "_" + str(int(idx)) + "(int flag, int tile_idx, INTERFACE_WIDTH* local_SP, volatile INTERFACE_WIDTH* searchSpace)" + '\n')
        _core_krnl_file.append("{" + '\n')
        _core_krnl_file.append("#pragma HLS INLINE OFF" + '\n')
        _core_krnl_file.append("	if (flag){" + '\n')
        _core_krnl_file.append("		for (int i(0); i<SP_LEN; ++i){" + '\n')
        _core_krnl_file.append("		#pragma HLS PIPELINE II=1" + '\n')
        _core_krnl_file.append("			local_SP[i] = searchSpace[QUERY_DATA_RESERVE + tile_idx*SP_LEN + i];" + '\n')
        _core_krnl_file.append("		}" + '\n')
        _core_krnl_file.append("	}" + '\n')
        _core_krnl_file.append("}" + '\n\n')

def GenerateCompute(_core_krnl_file, _parallel_sort, _SLR_name):
    if (_parallel_sort == 1):
        _core_krnl_file.append('void compute_' + _SLR_name + '(int flag, float* local_Query, INTERFACE_WIDTH* local_SP,' + '\n')
        _core_krnl_file.append('        float local_distance[PARALLEL_SORT_FACTOR][PARALLEL_SORT_SIZE+TOP])' + '\n')
        _core_krnl_file.append('{' + '\n')
        _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
        _core_krnl_file.append('	if (flag){' + '\n')
        _core_krnl_file.append('		int SP_idx = 0;' + '\n')
        _core_krnl_file.append('		for (int ii = 0 ; ii < PARALLEL_SORT_FACTOR; ++ii){' + '\n')
        _core_krnl_file.append('			for (int jj = 0; jj < PARALLEL_SORT_SIZE/FACTOR_W; ++jj){' + '\n')
        _core_krnl_file.append('#pragma HLS PIPELINE II=1' + '\n')
        _core_krnl_file.append('				SP_idx = ii * PARALLEL_SORT_SIZE/FACTOR_W + jj;' + '\n')
        _core_krnl_file.append('				for (int kk = 0; kk < FACTOR_W; ++kk){' + '\n')
        _core_krnl_file.append('					float delta_squared_sum = 0.0;' + '\n')
        _core_krnl_file.append('					int start_idx = kk * INPUT_DIM;' + '\n')
        _core_krnl_file.append('					for (int ll = 0; ll < INPUT_DIM; ++ll){' + '\n')
        _core_krnl_file.append('						unsigned int range_idx = (start_idx + ll) * 32;' + '\n')
        _core_krnl_file.append('						uint32_t sp_dim_item = local_SP[SP_idx].range(range_idx+31, range_idx);' + '\n')
        _core_krnl_file.append('						float sp_dim_item_value = *((float*)(&sp_dim_item));' + '\n')
        _core_krnl_file.append('#if DISTANCE_METRIC == 0 // manhattan' + '\n')
        _core_krnl_file.append('						float delta = abs(sp_dim_item_value - local_Query[ll]);' + '\n')
        _core_krnl_file.append('						delta_squared_sum += delta;' + '\n')
        _core_krnl_file.append('#elif DISTANCE_METRIC == 1 // L2' + '\n')
        _core_krnl_file.append('						float delta = sp_dim_item_value - local_Query[ll];' + '\n')
        _core_krnl_file.append('						delta_squared_sum += delta * delta;' + '\n')
        _core_krnl_file.append('#endif' + '\n')
        _core_krnl_file.append('					}' + '\n')
        _core_krnl_file.append('					local_distance[ii][jj*FACTOR_W+kk] = delta_squared_sum;' + '\n')
        _core_krnl_file.append('				}' + '\n')
        _core_krnl_file.append('			}' + '\n')
        _core_krnl_file.append('		}        ' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('}' + '\n\n')
    elif (_parallel_sort == 0):
        _core_krnl_file.append('void compute_' + _SLR_name + '(int flag, float* local_Query, INTERFACE_WIDTH* local_SP,' + '\n')
        _core_krnl_file.append('        float local_distance[DIS_LEN+TOP])' + '\n')
        _core_krnl_file.append('{' + '\n')
        _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
        _core_krnl_file.append('	if (flag){' + '\n')
        _core_krnl_file.append('        for (int i = 0; i < DIS_LEN; ++i) {' + '\n')
        _core_krnl_file.append('        #pragma HLS PIPELINE II=DIS_CALCULATION_II' + '\n')
        _core_krnl_file.append('            float delta_squared_sum = 0.0;' + '\n')
        _core_krnl_file.append('            for (int j = 0; j < FACTOR_W; ++j) {' + '\n')
        _core_krnl_file.append('                unsigned int search_space_idx = i * FACTOR_W + j;' + '\n')
        _core_krnl_file.append('                unsigned int feature_start_idx = j * NUM_FEATURES_PER_READ;' + '\n')
        _core_krnl_file.append('                for (int k = 0; k < NUM_FEATURES_PER_READ; ++k){' + '\n')
        _core_krnl_file.append('                    unsigned int range_idx = k * 32;' + '\n')
        _core_krnl_file.append('                    uint32_t sp_dim_item = local_SP[search_space_idx].range(range_idx+31, range_idx);' + '\n')
        _core_krnl_file.append('                    float sp_dim_item_value = *((float*)(&sp_dim_item));' + '\n')
        _core_krnl_file.append('#if DISTANCE_METRIC == 0 // manhattan' + '\n')
        _core_krnl_file.append('						float delta = abs(sp_dim_item_value - local_Query[feature_start_idx+k]);' + '\n')
        _core_krnl_file.append('						delta_squared_sum += delta;' + '\n')
        _core_krnl_file.append('#elif DISTANCE_METRIC == 1 // L2' + '\n')
        _core_krnl_file.append('						float delta = sp_dim_item_value - local_Query[feature_start_idx+k];' + '\n')
        _core_krnl_file.append('						delta_squared_sum += delta * delta;' + '\n')
        _core_krnl_file.append('#endif   ' + '\n')
        _core_krnl_file.append('                }' + '\n')
        _core_krnl_file.append('            }' + '\n')
        _core_krnl_file.append('            local_distance[i] = delta_squared_sum;' + '\n')
        _core_krnl_file.append('        }      ' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('}' + '\n\n')

def GenerateSort(_core_krnl_file, _parallel_sort, _SLR_name):
    _core_krnl_file.append('void swap_' + _SLR_name + '(float* a, float* b, int* x, int* y)' + '\n')
    _core_krnl_file.append('{' + '\n')
    _core_krnl_file.append('	float tmp_1;' + '\n')
    _core_krnl_file.append('	int tmp_2;' + '\n\n')
    _core_krnl_file.append('	tmp_1 = *a;' + '\n')
    _core_krnl_file.append('	*a = *b;' + '\n')
    _core_krnl_file.append('	*b = tmp_1;' + '\n\n')
    _core_krnl_file.append('	tmp_2 = *x;' + '\n')
    _core_krnl_file.append('	*x = *y;' + '\n')
    _core_krnl_file.append('	*y = tmp_2;' + '\n')
    _core_krnl_file.append('}' + '\n\n')

    if (_parallel_sort == 1):
        _core_krnl_file.append('void para_partial_sort_' + _SLR_name + '(float* local_distance, int start_id, float* local_kNearstDist, int* local_kNearstId)' + '\n')
        _core_krnl_file.append('{' + '\n')
        _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
        _core_krnl_file.append('    for (int i = 0; i < PARALLEL_SORT_SIZE+TOP; ++i) {' + '\n')
        _core_krnl_file.append('	#pragma HLS PIPELINE II=1' + '\n')
        _core_krnl_file.append('		local_kNearstDist[0] = local_distance[i];' + '\n')
        _core_krnl_file.append('		local_kNearstId[0] = start_id + i;' + '\n')
        _core_krnl_file.append('		//compare and swap odd' + '\n')
        _core_krnl_file.append('		for(int ii=1; ii<TOP+1; ii+=2){' + '\n')
        _core_krnl_file.append('		#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('		#pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + '\n')
        _core_krnl_file.append('		#pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + '\n')
        _core_krnl_file.append('			if(local_kNearstDist[ii] < local_kNearstDist[ii+1]){' + '\n')
        _core_krnl_file.append('				swap_' + _SLR_name + '(&local_kNearstDist[ii], &local_kNearstDist[ii+1], &local_kNearstId[ii], &local_kNearstId[ii+1]);' + '\n')
        _core_krnl_file.append('		    }' + '\n')
        _core_krnl_file.append('		}' + '\n')
        _core_krnl_file.append('		//compare and swap even' + '\n')
        _core_krnl_file.append('		for(int ii=1; ii<TOP+1; ii+=2){' + '\n')
        _core_krnl_file.append('		#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('		#pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + '\n')
        _core_krnl_file.append('		#pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + '\n')
        _core_krnl_file.append('			if(local_kNearstDist[ii] > local_kNearstDist[ii-1]){' + '\n')
        _core_krnl_file.append('				swap_' + _SLR_name + '(&local_kNearstDist[ii], &local_kNearstDist[ii-1], &local_kNearstId[ii], &local_kNearstId[ii-1]);' + '\n')
        _core_krnl_file.append('			}' + '\n')
        _core_krnl_file.append('		}' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('}' + '\n\n')
        _core_krnl_file.append('void sort_' + _SLR_name + '(int flag, int start_id, float local_distance[PARALLEL_SORT_FACTOR][PARALLEL_SORT_SIZE+TOP],' + '\n')
        _core_krnl_file.append('		  float local_kNearstDist_partial [PARALLEL_SORT_FACTOR][TOP+2], ' + '\n')
        _core_krnl_file.append('          int local_kNearstId_partial [PARALLEL_SORT_FACTOR][TOP+2])' + '\n')
        _core_krnl_file.append('{' + '\n')
        _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
        _core_krnl_file.append('	if (flag){' + '\n')
        _core_krnl_file.append('		int starting_id[PARALLEL_SORT_FACTOR];' + '\n')
        _core_krnl_file.append('		#pragma HLS ARRAY_PARTITION variable=starting_id complete dim=0' + '\n\n')
        _core_krnl_file.append('		for (int i = 0; i < PARALLEL_SORT_FACTOR; ++i){' + '\n')
        _core_krnl_file.append('		#pragma HLS PIPELINE II=1' + '\n')
        _core_krnl_file.append('			starting_id[i] = start_id+i*PARALLEL_SORT_SIZE;' + '\n')
        _core_krnl_file.append('		}' + '\n')
        _core_krnl_file.append('        ' + '\n')
        _core_krnl_file.append('		for (int i = 0; i < PARALLEL_SORT_FACTOR * PARALLEL_SORT_SIZE - DIS_LEN; i+=FACTOR_W){' + '\n')
        _core_krnl_file.append('		#pragma HLS PIPELINE II=1' + '\n')
        _core_krnl_file.append('			for (int j = 0; j < FACTOR_W; ++j){' + '\n')
        _core_krnl_file.append('			#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('                local_distance[PARALLEL_SORT_FACTOR-1][START_OF_PADDING+i+j] = MAX_FLT;' + '\n')
        _core_krnl_file.append('			}' + '\n')
        _core_krnl_file.append('		}        ' + '\n\n')
        _core_krnl_file.append('		for (int i = 0; i < PARALLEL_SORT_FACTOR; ++i){' + '\n')
        _core_krnl_file.append('		#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('			para_partial_sort_' + _SLR_name + '(local_distance[i], starting_id[i], local_kNearstDist_partial[i], local_kNearstId_partial[i]);' + '\n')
        _core_krnl_file.append('		}' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('	else{' + '\n')
        _core_krnl_file.append('		for (int i = 0; i < PARALLEL_SORT_FACTOR; ++i){' + '\n')
        _core_krnl_file.append('		#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('			for (int j = 0; j < TOP+2; ++j){' + '\n')
        _core_krnl_file.append('			#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('				local_kNearstDist_partial[i][j] = MAX_FLT;' + '\n')
        _core_krnl_file.append('				local_kNearstId_partial[i][j] = -1;' + '\n')
        _core_krnl_file.append('			}' + '\n')
        _core_krnl_file.append('		}' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('}' + '\n\n')
        _core_krnl_file.append('void merge_' + _SLR_name + '(float local_kNearstDist_partial [PARALLEL_SORT_FACTOR][TOP+2],' + '\n')
        _core_krnl_file.append('			int local_kNearstId_partial [PARALLEL_SORT_FACTOR][TOP+2], ' + '\n')
        _core_krnl_file.append('            float dist[TOP+2], int id[TOP+2])' + '\n')
        _core_krnl_file.append('{' + '\n')
        _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
        _core_krnl_file.append('	int idx[PARALLEL_SORT_FACTOR];' + '\n')
        _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=idx complete dim=0' + '\n')
        _core_krnl_file.append('    ' + '\n')
        _core_krnl_file.append('	for (int i = 0; i < PARALLEL_SORT_FACTOR; ++i){' + '\n')
        _core_krnl_file.append('	#pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('		idx[i] = TOP;' + '\n')
        _core_krnl_file.append('	}' + '\n\n')
        _core_krnl_file.append('    for (int i = TOP; i > 0; --i){' + '\n')
        _core_krnl_file.append('        float min_value = MAX_FLT;' + '\n')
        _core_krnl_file.append('        int min_idx = -1;' + '\n')
        _core_krnl_file.append('        for (int j = 0; j < PARALLEL_SORT_FACTOR; ++j){' + '\n')
        _core_krnl_file.append('        #pragma HLS PIPELINE II=1' + '\n')
        _core_krnl_file.append('            if (local_kNearstDist_partial[j][idx[j]] < min_value){' + '\n')
        _core_krnl_file.append('                min_value = local_kNearstDist_partial[j][idx[j]];' + '\n')
        _core_krnl_file.append('                min_idx = j;' + '\n')
        _core_krnl_file.append('            }' + '\n')
        _core_krnl_file.append('        }' + '\n')
        _core_krnl_file.append('        dist[i] = min_value;' + '\n')
        _core_krnl_file.append('        id[i] = local_kNearstId_partial[min_idx][idx[min_idx]];' + '\n\n')
        _core_krnl_file.append('        idx[min_idx] = idx[min_idx] - 1;' + '\n')
        _core_krnl_file.append('    }' + '\n')
        _core_krnl_file.append('}' + '\n\n')
    elif (_parallel_sort == 0):
        _core_krnl_file.append('void sort_' + _SLR_name + '(int flag, int start_id, float local_distance[DIS_LEN+TOP],' + '\n')
        _core_krnl_file.append('		  float local_kNearstDist[TOP+2], int local_kNearstId[TOP+2])' + '\n')
        _core_krnl_file.append('{' + '\n')
        _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
        _core_krnl_file.append('	if (flag){' + '\n')
        _core_krnl_file.append('        for (int i = 0; i < DIS_LEN+TOP; ++i) {' + '\n')
        _core_krnl_file.append('        #pragma HLS PIPELINE II=1' + '\n')
        _core_krnl_file.append('            local_kNearstDist[0] = local_distance[i];' + '\n')
        _core_krnl_file.append('            local_kNearstId[0] = start_id + i;' + '\n')
        _core_krnl_file.append('            //compare and swap odd' + '\n')
        _core_krnl_file.append('            for(int ii=1; ii<TOP+1; ii+=2){' + '\n')
        _core_krnl_file.append('            #pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + '\n')
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + '\n')
        _core_krnl_file.append('                if(local_kNearstDist[ii] < local_kNearstDist[ii+1]){' + '\n')
        _core_krnl_file.append('                    swap_' + _SLR_name + '(&local_kNearstDist[ii], &local_kNearstDist[ii+1], &local_kNearstId[ii], &local_kNearstId[ii+1]);' + '\n')
        _core_krnl_file.append('                }' + '\n')
        _core_krnl_file.append('            }' + '\n')
        _core_krnl_file.append('            //compare and swap even' + '\n')
        _core_krnl_file.append('            for(int ii=1; ii<TOP+1; ii+=2){' + '\n')
        _core_krnl_file.append('            #pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + '\n')
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + '\n')
        _core_krnl_file.append('                if(local_kNearstDist[ii] > local_kNearstDist[ii-1]){' + '\n')
        _core_krnl_file.append('                    swap_' + _SLR_name + '(&local_kNearstDist[ii], &local_kNearstDist[ii-1], &local_kNearstId[ii], &local_kNearstId[ii-1]);' + '\n')
        _core_krnl_file.append('                }' + '\n')
        _core_krnl_file.append('            }' + '\n')
        _core_krnl_file.append('        }' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('	else{' + '\n')
        _core_krnl_file.append('        for (int i = 0; i < TOP+2; ++i){' + '\n')
        _core_krnl_file.append('        #pragma HLS UNROLL' + '\n')
        _core_krnl_file.append('            local_kNearstDist[i] = MAX_FLT;' + '\n')
        _core_krnl_file.append('            local_kNearstId[i] = -1;' + '\n')
        _core_krnl_file.append('        }' + '\n')
        _core_krnl_file.append('	}' + '\n')
        _core_krnl_file.append('}' + '\n')

def GenerateGlobalMerge(_core_krnl_file, _SLR_name):
    _core_krnl_file.append('void merge_global_' + _SLR_name + '(float local_kNearstDist [NUM_PART][TOP+2], ' + '\n')
    _core_krnl_file.append('                  int local_kNearstId [NUM_PART][TOP+2], ' + '\n')
    _core_krnl_file.append('                  float dist[TOP+2], int id[TOP+2])' + '\n')
    _core_krnl_file.append('{' + '\n')
    _core_krnl_file.append('#pragma HLS INLINE OFF' + '\n')
    _core_krnl_file.append('	int idx[NUM_PART];' + '\n')
    _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=idx complete dim=0' + '\n')
    _core_krnl_file.append('    ' + '\n')
    _core_krnl_file.append('	for (int i = 0; i < NUM_PART; ++i){' + '\n')
    _core_krnl_file.append('	#pragma HLS UNROLL' + '\n')
    _core_krnl_file.append('		idx[i] = TOP;' + '\n')
    _core_krnl_file.append('	}' + '\n\n')
    _core_krnl_file.append('    for (int i = TOP; i > 0; --i){' + '\n')
    _core_krnl_file.append('        float min_value = MAX_FLT;' + '\n')
    _core_krnl_file.append('        int min_idx = -1;' + '\n')
    _core_krnl_file.append('        for (int j = 0; j < NUM_PART; ++j){' + '\n')
    _core_krnl_file.append('        #pragma HLS PIPELINE II=1' + '\n')
    _core_krnl_file.append('            if (local_kNearstDist[j][idx[j]] < min_value){' + '\n')
    _core_krnl_file.append('                min_value = local_kNearstDist[j][idx[j]];' + '\n')
    _core_krnl_file.append('                min_idx = j;' + '\n')
    _core_krnl_file.append('            }' + '\n')
    _core_krnl_file.append('        }' + '\n')
    _core_krnl_file.append('        dist[i] = min_value;' + '\n')
    _core_krnl_file.append('        id[i] = local_kNearstId[min_idx][idx[min_idx]];' + '\n\n')    
    _core_krnl_file.append('        idx[min_idx] = idx[min_idx] - 1;' + '\n')
    _core_krnl_file.append('    }' + '\n')
    _core_krnl_file.append('}' + '\n')

def GenerateTopLevel(_core_krnl_file, _parallel_sort, _num_PE, _krnl_top_name, _SLR_name):
    _core_krnl_file.append('void ' + _krnl_top_name + '(' + '\n')
    for idx in range (_num_PE):
        _core_krnl_file.append('    INTERFACE_WIDTH* searchSpace_' + str(idx) + ',' + '\n') 
        _core_krnl_file.append('    int start_id_' + str(idx) + ',' + '\n') 
    _core_krnl_file.append('    hls::stream<pkt> &out)' + '\n')
    _core_krnl_file.append('{' + '\n')
    for idx in range (_num_PE):
        _core_krnl_file.append('#pragma HLS INTERFACE m_axi port=searchSpace_' + str(idx) + ' offset=slave bundle=gmem' + str(idx) + '\n')
        _core_krnl_file.append('#pragma HLS INTERFACE s_axilite port=searchSpace_' + str(idx) + ' bundle=control' + '\n')
        _core_krnl_file.append('#pragma HLS INTERFACE s_axilite port=start_id_' + str(idx) + ' bundle=control' + '\n')
    _core_krnl_file.append('#pragma HLS INTERFACE axis port=out' + '\n')
    _core_krnl_file.append('#pragma HLS INTERFACE s_axilite port=return bundle=control' + '\n\n')

    for idx in range (_num_PE):
        _core_krnl_file.append('    float local_Query_' + str(idx) + '[INPUT_DIM];' + '\n')
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_Query_' + str(idx) + ' complete dim=1' + '\n')
        _core_krnl_file.append('    INTERFACE_WIDTH local_SP_' + str(idx) + '_A[SP_LEN];' + '\n')
        _core_krnl_file.append('    #pragma HLS RESOURCE variable=local_SP_' + str(idx) + '_A core=XPM_MEMORY uram' + '\n')
        _core_krnl_file.append('    INTERFACE_WIDTH local_SP_' + str(idx) + '_B[SP_LEN];' + '\n')
        _core_krnl_file.append('    #pragma HLS RESOURCE variable=local_SP_' + str(idx) + '_B core=XPM_MEMORY uram' + '\n\n')

    if (_parallel_sort == 1):
        for idx in range (_num_PE):
            _core_krnl_file.append('    float local_distance_' + str(idx) + '_A[PARALLEL_SORT_FACTOR][PARALLEL_SORT_SIZE+TOP];' + '\n')
            _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_distance_' + str(idx) + '_A complete dim=1' + '\n')
            _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_distance_' + str(idx) + '_A cyclic factor=FACTOR_W dim=2' + '\n')
            _core_krnl_file.append('    float local_distance_' + str(idx) + '_B[PARALLEL_SORT_FACTOR][PARALLEL_SORT_SIZE+TOP];' + '\n')
            _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_distance_' + str(idx) + '_B complete dim=1' + '\n')
            _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_distance_' + str(idx) + '_B cyclic factor=FACTOR_W dim=2' + '\n')
            _core_krnl_file.append('	float local_kNearstDist_partial_' + str(idx) + '[PARALLEL_SORT_FACTOR][(TOP+2)];' + '\n')
            _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_kNearstDist_partial_' + str(idx) + ' complete dim=0' + '\n')
            _core_krnl_file.append('	int local_kNearstId_partial_' + str(idx) + '[PARALLEL_SORT_FACTOR][(TOP+2)];' + '\n')
            _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_kNearstId_partial_' + str(idx) + ' complete dim=0' + '\n\n')
    elif (_parallel_sort == 0):
        for idx in range (_num_PE):
            _core_krnl_file.append('    float local_distance_' + str(idx) + '_A[DIS_LEN+TOP];' + '\n')
            _core_krnl_file.append('    float local_distance_' + str(idx) + '_B[DIS_LEN+TOP];' + '\n\n')

    _core_krnl_file.append('	float local_kNearstDist[NUM_PART][TOP+2];' + '\n')
    _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_kNearstDist complete dim=0' + '\n')
    _core_krnl_file.append('	int local_kNearstId[NUM_PART][TOP+2];' + '\n')
    _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=local_kNearstId complete dim=0' + '\n\n')
    _core_krnl_file.append('	float global_kNearstDist[TOP+2];' + '\n')
    _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=global_kNearstDist complete' + '\n')
    _core_krnl_file.append('	int global_kNearstId[TOP+2];' + '\n')
    _core_krnl_file.append('	#pragma HLS ARRAY_PARTITION variable=global_kNearstId complete' + '\n')

    _core_krnl_file.append('	LOAD_QUERY: for (int i = 0; i < INPUT_DIM; ++i){' + '\n')
    _core_krnl_file.append('        int input_rd_idx = i / NUM_FEATURES_PER_READ;' + '\n')
    _core_krnl_file.append('        int range_idx = i % NUM_FEATURES_PER_READ;' + '\n\n')
    for idx in range (_num_PE):
        _core_krnl_file.append('        uint32_t sp_dim_item_' + str(idx) + ' = searchSpace_' + str(idx) + '[input_rd_idx].range(range_idx*32+31, range_idx*32);' + '\n')
        _core_krnl_file.append('        local_Query_' + str(idx) + '[i] = *((float*)(&sp_dim_item_' + str(idx) + '));' + '\n')
    _core_krnl_file.append('    }' + '\n\n')

    _core_krnl_file.append('	ITERATION_LOOP: for (int it_idx = 0; it_idx < NUM_ITERATIONS; ++it_idx)' + '\n')
    _core_krnl_file.append('	{' + '\n')
    if (_parallel_sort == 1):
        _core_krnl_file.append('        for (int i = 0; i < PARALLEL_SORT_FACTOR; ++i){' + '\n')
        _core_krnl_file.append('            for (int j = 0; j < TOP; ++j){' + '\n')
        _core_krnl_file.append('            #pragma HLS PIPELINE II=1' + '\n')
        for idx in range (_num_PE):
            _core_krnl_file.append('                local_distance_' + str(idx) + '_A[i][PARALLEL_SORT_SIZE+j] = MAX_FLT;' + '\n')
            _core_krnl_file.append('                local_distance_' + str(idx) + '_B[i][PARALLEL_SORT_SIZE+j] = MAX_FLT;' + '\n')
        _core_krnl_file.append('            }' + '\n')
        _core_krnl_file.append('        }' + '\n\n')
    elif (_parallel_sort == 0):
        _core_krnl_file.append('        for (int i = 0; i < TOP; ++i){' + '\n')
        _core_krnl_file.append('        #pragma HLS PIPELINE II=1' + '\n')
        for idx in range (_num_PE):
            _core_krnl_file.append('            local_distance_' + str(idx) + '_A[DIS_LEN+i] = MAX_FLT;' + '\n')
            _core_krnl_file.append('            local_distance_' + str(idx) + '_B[DIS_LEN+i] = MAX_FLT;' + '\n')
        _core_krnl_file.append('        }' + '\n\n')

    _core_krnl_file.append('		for(int i = 0; i < NUM_OF_TILES+2; ++i){' + '\n')
    _core_krnl_file.append('			int load_img_flag = i >= 0 && i < NUM_OF_TILES;' + '\n')
    _core_krnl_file.append('			int compute_flag = i >= 1 && i < NUM_OF_TILES + 1;' + '\n')
    _core_krnl_file.append('			int sort_flag = i >= 2 && i < NUM_OF_TILES + 2;' + '\n')
    _core_krnl_file.append('			if (i % 2 == 0) {' + '\n')

    for idx in range (_num_PE):
        _core_krnl_file.append('				load_' + _SLR_name + '_' + str(idx) + '(load_img_flag, i, local_SP_' + str(idx) + '_A, searchSpace_' + str(idx) + ');' + '\n')

    for idx in range (_num_PE):
        _core_krnl_file.append('				compute_' + _SLR_name + '(compute_flag, local_Query_' + str(idx) + ', local_SP_' + str(idx) + '_B, local_distance_' + str(idx) + '_B);' + '\n')

    if (_parallel_sort == 1):
        for idx in range (_num_PE):
            _core_krnl_file.append('				sort_' + _SLR_name + '(sort_flag, start_id_' + str(idx) + '+(i-2)*DIS_LEN, local_distance_' + str(idx) + '_A, local_kNearstDist_partial_' + str(idx) + ', local_kNearstId_partial_' + str(idx) + ');' + '\n')
    elif (_parallel_sort == 0):
        for idx in range (_num_PE):
            _core_krnl_file.append('				sort_' + _SLR_name + '(sort_flag, start_id_' + str(idx) + '+(i-2)*DIS_LEN, local_distance_' + str(idx) + '_A, local_kNearstDist[' + str(idx) + '], local_kNearstId[' + str(idx) + ']);' + '\n')
    
    _core_krnl_file.append('			}' + '\n')
    _core_krnl_file.append('			else {' + '\n')
    
    for idx in range (_num_PE):
        _core_krnl_file.append('				load_' + _SLR_name + '_' + str(idx) + '(load_img_flag, i, local_SP_' + str(idx) + '_B, searchSpace_' + str(idx) + ');' + '\n')
    for idx in range (_num_PE):
        _core_krnl_file.append('				compute_' + _SLR_name + '(compute_flag, local_Query_' + str(idx) + ', local_SP_' + str(idx) + '_A, local_distance_' + str(idx) + '_A);' + '\n')

    if (_parallel_sort == 1):
        for idx in range (_num_PE):
            _core_krnl_file.append('                sort_' + _SLR_name + '(sort_flag, start_id_' + str(idx) + '+(i-2)*DIS_LEN, local_distance_' + str(idx) + '_B, local_kNearstDist_partial_' + str(idx) + ', local_kNearstId_partial_' + str(idx) + ');' + '\n')
    elif (_parallel_sort == 0):
        for idx in range (_num_PE):
            _core_krnl_file.append('                sort_' + _SLR_name + '(sort_flag, start_id_' + str(idx) + '+(i-2)*DIS_LEN, local_distance_' + str(idx) + '_B, local_kNearstDist[' + str(idx) + '], local_kNearstId[' + str(idx) + ']);' + '\n')

    _core_krnl_file.append('			}' + '\n')
    _core_krnl_file.append('		}' + '\n')

    if (_parallel_sort == 1):
        for idx in range (_num_PE):
            _core_krnl_file.append('        merge_' + _SLR_name + '(local_kNearstDist_partial_' + str(idx) + ', local_kNearstId_partial_' + str(idx) + ', local_kNearstDist[' + str(idx) + '], local_kNearstId[' + str(idx) + ']);' + '\n')
    
    if (_num_PE > 1):
        _core_krnl_file.append('        merge_global_' + _SLR_name + '(local_kNearstDist, local_kNearstId, global_kNearstDist, global_kNearstId);' + '\n')
    else:
        _core_krnl_file.append('		for (int j = 0; j < TOP+2; ++j){' + '\n')
        _core_krnl_file.append('			global_kNearstDist[j] = local_kNearstDist[0][j];' + '\n')
        _core_krnl_file.append('			global_kNearstId[j] = local_kNearstId[0][j];' + '\n')
        _core_krnl_file.append('		}        ' + '\n')

    _core_krnl_file.append('	}' + '\n\n')
    _core_krnl_file.append('    STREAM_WIDTH v_data;' + '\n')
    _core_krnl_file.append('    float temp_data;' + '\n')
    _core_krnl_file.append('	DIST_OUT: for (int i = 1; i < TOP+1; ++i){' + '\n')
    _core_krnl_file.append('	#pragma HLS PIPELINE II=1' + '\n')
    _core_krnl_file.append('        temp_data = global_kNearstDist[i]; ' + '\n')
    _core_krnl_file.append('        v_data.range(31, 0) = *((uint32_t *)(&temp_data));' + '\n')
    _core_krnl_file.append('        pkt v;' + '\n')
    _core_krnl_file.append('		v.data = v_data;' + '\n')
    _core_krnl_file.append('		out.write(v);' + '\n')
    _core_krnl_file.append('	}' + '\n')
    _core_krnl_file.append('	ID_OUT: for (int i = 1; i < TOP+1; ++i){' + '\n')
    _core_krnl_file.append('	#pragma HLS PIPELINE II=1' + '\n')
    _core_krnl_file.append('		pkt v_id;' + '\n')
    _core_krnl_file.append('		v_id.data = global_kNearstId[i]; ' + '\n')
    _core_krnl_file.append('		out.write(v_id);' + '\n')
    _core_krnl_file.append('	}' + '\n')
    _core_krnl_file.append('	return;' + '\n')
    _core_krnl_file.append('}' + '\n')
    _core_krnl_file.append('}' + '\n')

def Generate_GlobalSort_Design(_slr_num_pe):
    sort_file_name = 'krnl_globalSort.cpp'
    sort_file = []

    sort_file.append('#include "krnl_config.h"' + '\n')
    sort_file.append('extern "C" {' + '\n')
    sort_file.append('void seq_global_merge(float local_kNearstDist_partial[NUM_USED_SLR][TOP],' + '\n')
    sort_file.append('						int local_kNearstId_partial[NUM_USED_SLR][TOP], ' + '\n')
    sort_file.append('						float* dist, int* id)' + '\n')
    sort_file.append('{' + '\n')
    sort_file.append('#pragma HLS INLINE OFF' + '\n')
    sort_file.append('	int idx[NUM_USED_SLR];' + '\n')
    sort_file.append('	#pragma HLS ARRAY_PARTITION variable=idx complete dim=0' + '\n')
    sort_file.append('	for (int i = 0; i < NUM_USED_SLR; ++i){' + '\n')
    sort_file.append('	#pragma HLS UNROLL' + '\n')
    sort_file.append('		idx[i] = TOP-1;' + '\n')
    sort_file.append('	}' + '\n')
    sort_file.append('	for (int i = TOP-1; i >= 0; --i){' + '\n')
    sort_file.append('		float min_value = MAX_FLT;' + '\n')
    sort_file.append('		int min_idx = -1;' + '\n')
    sort_file.append('		for (int j = 0; j < NUM_USED_SLR; ++j){' + '\n')
    sort_file.append('		#pragma HLS PIPELINE II=1' + '\n')
    sort_file.append('			if (local_kNearstDist_partial[j][idx[j]] < min_value){' + '\n')
    sort_file.append('				min_value = local_kNearstDist_partial[j][idx[j]];' + '\n')
    sort_file.append('				min_idx = j;' + '\n')
    sort_file.append('			}' + '\n')
    sort_file.append('		}' + '\n')
    sort_file.append('		dist[i] = min_value;' + '\n')
    sort_file.append('		id[i] = local_kNearstId_partial[min_idx][idx[min_idx]];' + '\n')
    sort_file.append('		idx[min_idx] = idx[min_idx] - 1;' + '\n')
    sort_file.append('	}' + '\n')
    sort_file.append('}' + '\n\n')

    sort_file.append('void krnl_globalSort(' + '\n')
    input_idx = 0
    for i in range (len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            sort_file.append('    hls::stream<pkt> &in' + str(input_idx) + ',    // Internal Stream' + '\n')
            input_idx += 1
    sort_file.append('    float *output_knnDist,    // Output Result' + '\n')
    sort_file.append('    int *output_knnId         // Output Result' + '\n')
    sort_file.append(') {' + '\n')
    input_idx = 0
    for i in range (len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            sort_file.append('#pragma HLS INTERFACE axis port = in' + str(input_idx) + '\n')
            input_idx += 1
    sort_file.append('#pragma HLS INTERFACE m_axi port=output_knnDist offset=slave bundle=gmem1' + '\n')
    sort_file.append('#pragma HLS INTERFACE s_axilite port=output_knnDist bundle=control' + '\n')
    sort_file.append('#pragma HLS INTERFACE m_axi port=output_knnId offset=slave bundle=gmem1' + '\n')
    sort_file.append('#pragma HLS INTERFACE s_axilite port=output_knnId bundle=control' + '\n')
    sort_file.append('#pragma HLS INTERFACE s_axilite port=return bundle=control' + '\n\n')

    sort_file.append('	float local_kNearstDist_partial[NUM_USED_SLR][TOP];' + '\n')
    sort_file.append('	#pragma HLS ARRAY_PARTITION variable=local_kNearstDist_partial complete dim=0' + '\n')
    sort_file.append('	int local_kNearstId_partial[NUM_USED_SLR][TOP];' + '\n')
    sort_file.append('	#pragma HLS ARRAY_PARTITION variable=local_kNearstId_partial complete dim=0' + '\n\n')
    sort_file.append('	float output_dist[TOP];' + '\n')
    sort_file.append('	#pragma HLS ARRAY_PARTITION variable=output_dist complete' + '\n')
    sort_file.append('	int output_id[TOP];' + '\n')
    sort_file.append('	#pragma HLS ARRAY_PARTITION variable=output_id complete' + '\n\n')

    sort_file.append('    for (unsigned int i=0; i<TOP; ++i){' + '\n')
    sort_file.append('#pragma HLS PIPELINE II=1' + '\n')
    input_idx = 0
    for i in range (len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            sort_file.append('      pkt v' + str(input_idx) + ' = in' + str(input_idx) + '.read();' + '\n')
            sort_file.append('      uint32_t v' + str(input_idx) + '_item = v' + str(input_idx) + '.data.range(31, 0);' + '\n')
            sort_file.append('      local_kNearstDist_partial[' + str(input_idx) + '][i] = *((float*)(&v' + str(input_idx) + '_item));' + '\n')
            input_idx += 1
    sort_file.append('  }' + '\n\n')

    sort_file.append('  for (unsigned int i=0; i<TOP; ++i){' + '\n')
    sort_file.append('#pragma HLS PIPELINE II=1' + '\n')
    input_idx = 0
    for i in range (len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            sort_file.append('      pkt v' + str(input_idx) + '_id = in' + str(input_idx) + '.read();' + '\n')
            sort_file.append('      local_kNearstId_partial[' + str(input_idx) + '][i] = v' + str(input_idx) + '_id.data;' + '\n')
            input_idx += 1
    sort_file.append('  }' + '\n\n')

    sort_file.append('	seq_global_merge(local_kNearstDist_partial, local_kNearstId_partial, output_dist, output_id);' + '\n\n')
    sort_file.append('    for (unsigned int i=0; i<TOP; ++i){' + '\n')
    sort_file.append('#pragma HLS PIPELINE II=1' + '\n')
    sort_file.append('    	output_knnDist[i] = output_dist[i];' + '\n')
    sort_file.append('        output_knnId[i] = output_id[i];' + '\n')
    sort_file.append('    }' + '\n')
    sort_file.append('}' + '\n')
    sort_file.append('}' + '\n')

    with open(sort_file_name, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(sort_file)

def Generate_Host_Code(_memory_type, _num_mem_banks, _total_num_pe, _slr_num_pe):
    host_file_name = 'host.cpp'
    host_file = []

    host_file.append('#include <algorithm>' + '\n')
    host_file.append('#include <iostream>' + '\n')
    host_file.append('#include <stdint.h>' + '\n')
    host_file.append('#include <stdlib.h>' + '\n')
    host_file.append('#include <string.h>' + '\n')
    host_file.append('#include <vector>' + '\n')
    host_file.append('#include <ctime>' + '\n')
    host_file.append('' + '\n')
    host_file.append('// This extension file is required for stream APIs' + '\n')
    host_file.append('#include "CL/cl_ext_xilinx.h"' + '\n')
    host_file.append('#include "xcl2.hpp"' + '\n')
    host_file.append('#include "krnl_config.h"' + '\n')
    host_file.append('' + '\n')
    host_file.append('//HBM Banks requirements' + '\n')
    host_file.append('#define MAX_HBM_BANKCOUNT 32' + '\n')
    host_file.append('#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY' + '\n')
    host_file.append('const int bank[MAX_HBM_BANKCOUNT] = {' + '\n')
    host_file.append('    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),' + '\n')
    host_file.append('    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),' + '\n')
    host_file.append('    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),' + '\n')
    host_file.append('    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),' + '\n')
    host_file.append('    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),' + '\n')
    host_file.append('    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),' + '\n')
    host_file.append('    BANK_NAME(30), BANK_NAME(31)};' + '\n')
    host_file.append('// Function for verifying results' + '\n')
    host_file.append('bool verify(std::vector<float, aligned_allocator<float>> &sw_dist,' + '\n')
    host_file.append('            std::vector<float, aligned_allocator<float>> &hw_dist,' + '\n')
    host_file.append('            std::vector<int, aligned_allocator<int>> &sw_id,' + '\n')
    host_file.append('            std::vector<int, aligned_allocator<int>> &hw_id,' + '\n')
    host_file.append('            unsigned int size) ' + '\n')
    host_file.append('{' + '\n')
    host_file.append('    bool check = true;' + '\n')
    host_file.append('    int num_err = 0;' + '\n')
    host_file.append('    for (unsigned int i=0; i<size; ++i) {' + '\n')
    host_file.append('        if (sw_dist[i] != hw_dist[i] ||' + '\n')
    host_file.append('            sw_id[i] != hw_id[i]) {' + '\n')
    host_file.append('            check = false;' + '\n')
    host_file.append('            if (num_err < 10){' + '\n')
    host_file.append('                std::cout << "Error: Result mismatch"' + '\n')
    host_file.append('                          << std::endl;' + '\n')
    host_file.append('                std::cout << "i = " << i' + '\n')
    host_file.append('                          << " CPU result = " << sw_id[i] << " - " << sw_dist[i]' + '\n')
    host_file.append('                          << " FPGA result = " << hw_id[i] << " - " << hw_dist[i]' + '\n')
    host_file.append('                          << " Error delta = " << std::abs(sw_dist[i]-hw_dist[i])' + '\n')
    host_file.append('                          << std::endl;' + '\n')
    host_file.append('                num_err++;' + '\n')
    host_file.append('            }else{' + '\n')
    host_file.append('                break;' + '\n')
    host_file.append('            }' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    return check;' + '\n')
    host_file.append('}' + '\n')
    host_file.append('' + '\n')
    host_file.append('void Generate_sw_verif_data(std::vector<float, aligned_allocator<float>> &query,' + '\n')
    host_file.append('                            std::vector<float, aligned_allocator<float>> &searchSpace,' + '\n')
    host_file.append('                            std::vector<float, aligned_allocator<float>> &dist,' + '\n')
    host_file.append('                            std::vector<int, aligned_allocator<int>> &id,' + '\n')
    host_file.append('                            unsigned int num_of_points)' + '\n')
    host_file.append('{' + '\n')
    host_file.append('    // Generate random float data' + '\n')
    host_file.append('    std::fill(query.begin(), query.end(), 0.0);    ' + '\n')
    host_file.append('    std::fill(searchSpace.begin(), searchSpace.end(), 0.0);' + '\n')
    host_file.append('    std::fill(dist.begin(), dist.end(), 99999.9999);' + '\n')
    host_file.append('    std::fill(id.begin(), id.end(), 0);' + '\n')
    host_file.append('' + '\n')
    host_file.append('    for (unsigned int i=0; i<INPUT_DIM; ++i){' + '\n')
    host_file.append('        query[i] = static_cast <float>(rand()) / static_cast<float>(RAND_MAX);' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    for (unsigned int i=0; i<num_of_points*INPUT_DIM; ++i){' + '\n')
    host_file.append('        searchSpace[i] = static_cast <float>(rand()) / static_cast<float>(RAND_MAX);' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Calculate distance result' + '\n')
    host_file.append('    float delta_sum=0.0;' + '\n')
    host_file.append('    float delta=0.0;' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> distance(num_of_points);' + '\n')
    host_file.append('    for (unsigned int i=0; i<num_of_points; ++i){' + '\n')
    host_file.append('        delta_sum = 0.0;    ' + '\n')
    host_file.append('        for (unsigned int j=0; j<INPUT_DIM; ++j){' + '\n')
    host_file.append('            if (DISTANCE_METRIC == 0){' + '\n')
    host_file.append('                delta = abs(query[j] - searchSpace[i*INPUT_DIM+j]);' + '\n')
    host_file.append('                delta_sum += delta;' + '\n')
    host_file.append('            }else if (DISTANCE_METRIC == 1){' + '\n')
    host_file.append('                delta = query[j] - searchSpace[i*INPUT_DIM+j];' + '\n')
    host_file.append('                delta_sum += delta * delta;' + '\n')
    host_file.append('            }' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('        distance[i] = delta_sum;' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Sort distance' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> distance_cpy(distance);' + '\n')
    host_file.append('    std::sort(distance_cpy.begin(), distance_cpy.end());    ' + '\n')
    host_file.append('    for (int i(0); i<TOP; ++i){' + '\n')
    host_file.append('        dist[i] = distance_cpy[TOP-1-i];' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('' + '\n')
    host_file.append('    for (int i(0); i<TOP; ++i){' + '\n')
    host_file.append('        for (int j(0); j < num_of_points; ++j){' + '\n')
    host_file.append('            if (distance[j] == dist[i]){' + '\n')
    host_file.append('                id[i] = j;' + '\n')
    host_file.append('                break;' + '\n')
    host_file.append('            }' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('    }    ' + '\n')
    host_file.append('    ' + '\n')
    host_file.append('    return;' + '\n')
    host_file.append('}' + '\n\n')
    host_file.append('int main(int argc, char *argv[]) {' + '\n')
    host_file.append('    if (argc != 2) {' + '\n')
    host_file.append('        printf("Usage: %s <XCLBIN> \\n", argv[0]);' + '\n')
    host_file.append('        return -1;' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    std::srand(std::time(NULL));' + '\n')
    host_file.append('    std::string binaryFile = argv[1];' + '\n')
    host_file.append('    cl_int err;' + '\n')
    host_file.append('    cl::CommandQueue q;' + '\n')

    for i in range(len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            host_file.append('    cl::Kernel cmpt_krnl_SLR' + str(i) + ';' + '\n')
    
    host_file.append('    cl::Kernel aggregate_krnl;' + '\n')
    host_file.append('    cl::Context context;' + '\n')
    host_file.append('    auto devices = xcl::get_xil_devices();' + '\n')
    host_file.append('    auto fileBuf = xcl::read_binary_file(binaryFile);' + '\n')
    host_file.append('    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};' + '\n')
    host_file.append('    int valid_device = 0;' + '\n')
    host_file.append('    for (unsigned int i = 0; i < devices.size(); i++) {' + '\n')
    host_file.append('        auto device = devices[i];' + '\n')
    host_file.append('        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));' + '\n')
    host_file.append('        OCL_CHECK(err,' + '\n')
    host_file.append('                  q = cl::CommandQueue(context,' + '\n')
    host_file.append('                                       {device},' + '\n')
    host_file.append('                                       CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |' + '\n')
    host_file.append('                                           CL_QUEUE_PROFILING_ENABLE,' + '\n')
    host_file.append('                                       &err));' + '\n')
    host_file.append('        std::cout << "Trying to program device[" << i' + '\n')
    host_file.append('                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;' + '\n')
    host_file.append('        cl::Program program(context, {device}, bins, NULL, &err);' + '\n')
    host_file.append('        if (err != CL_SUCCESS) {' + '\n')
    host_file.append('            std::cout << "Failed to program device[" << i' + '\n')
    host_file.append('                      << "] with xclbin file!\\n";' + '\n')
    host_file.append('        } else {' + '\n')
    host_file.append('            std::cout << "Device[" << i << "]: program successful!\\n";' + '\n')
    host_file.append('            std::string krnl_name_full;' + '\n')
    for i in range(len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            host_file.append('            krnl_name_full = "krnl_partialKnn_SLR' + str(i) + ':{krnl_partialKnn_SLR' + str(i) + '_1}";' + '\n')
            host_file.append('            OCL_CHECK(err, cmpt_krnl_SLR' + str(i) + ' = cl::Kernel(program, krnl_name_full.c_str(), &err));' + '\n')
            host_file.append('            printf("Creating a kernel [%s] \\n", krnl_name_full.c_str());' + '\n')
    host_file.append('            krnl_name_full = "krnl_globalSort:{krnl_globalSort_1}";' + '\n')
    host_file.append('            OCL_CHECK(err, aggregate_krnl = cl::Kernel(program, krnl_name_full.c_str(), &err));' + '\n')
    host_file.append('            printf("Creating a kernel [%s] \\n", krnl_name_full.c_str());' + '\n')
    host_file.append('            valid_device++;' + '\n')
    host_file.append('            break;' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    if (valid_device == 0) {' + '\n')
    host_file.append('        std::cout << "Failed to program any device found, exit!\\n";' + '\n')
    host_file.append('        exit(EXIT_FAILURE);' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    int dataSize = NUM_SP_PTS_PADDED;' + '\n')
    host_file.append('    if (xcl::is_emulation()) {' + '\n')
    host_file.append('        dataSize = NUM_SP_PTS_PADDED;' + '\n')
    host_file.append('    }    ' + '\n')

    host_file.append('    const unsigned int QUERY_FEATURE_RESERVE = 128;' + '\n')
    host_file.append('    ' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> searchspace_data(dataSize*INPUT_DIM);' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> searchspace_data_part[NUM_KERNEL];' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> query_data(INPUT_DIM);' + '\n')
    host_file.append('' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> sw_dist(TOP);' + '\n')
    host_file.append('    std::vector<int, aligned_allocator<int>> sw_id(TOP);' + '\n')
    host_file.append('    std::vector<float, aligned_allocator<float>> hw_dist(TOP);' + '\n')
    host_file.append('    std::vector<int, aligned_allocator<int>> hw_id(TOP);' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Create the test data' + '\n')
    host_file.append('    Generate_sw_verif_data(query_data, searchspace_data, sw_dist, sw_id, dataSize);' + '\n')
    host_file.append('    // Partition the full matrix into separate submatrices' + '\n')
    host_file.append('    int starting_idx = 0;' + '\n')
    host_file.append('    int part_size = dataSize*INPUT_DIM/NUM_KERNEL;' + '\n')
    host_file.append('    for (int i = 0; i < NUM_KERNEL; ++i) {' + '\n')
    host_file.append('        starting_idx = i*part_size;' + '\n')
    host_file.append('        searchspace_data_part[i].resize(QUERY_FEATURE_RESERVE + part_size);' + '\n')
    host_file.append('        for (int j = 0; j < QUERY_FEATURE_RESERVE; ++j){' + '\n')
    host_file.append('            searchspace_data_part[i][j] = 0.0;' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('        for (int j = 0; j < INPUT_DIM; ++j){' + '\n')
    host_file.append('            searchspace_data_part[i][j] = query_data[j];' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('        for (int j = 0; j < part_size; ++j){' + '\n')
    host_file.append('            searchspace_data_part[i][QUERY_FEATURE_RESERVE + j] = searchspace_data[starting_idx+j];' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    ' + '\n')
    host_file.append('    // Initializing hw output vectors to zero' + '\n')
    host_file.append('    std::fill(hw_dist.begin(), hw_dist.end(), 0.0);' + '\n')
    host_file.append('    std::fill(hw_id.begin(), hw_id.end(), 0.0);' + '\n')
    host_file.append('' + '\n')
    host_file.append('    std::vector<cl_mem_ext_ptr_t> inputSearchSpaceBufExt(NUM_KERNEL);' + '\n')
    host_file.append('    cl_mem_ext_ptr_t outputResultDistBufExt;' + '\n')
    host_file.append('    cl_mem_ext_ptr_t outputResultIdBufExt;' + '\n')
    host_file.append('' + '\n')
    host_file.append('    std::vector<cl::Buffer> buffer_input_searchspace(NUM_KERNEL);' + '\n')
    host_file.append('    cl::Buffer buffer_output_dist_result;' + '\n')
    host_file.append('    cl::Buffer buffer_output_id_result;' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // For Allocating Buffer to specific Global Memory Bank, user has to use cl_mem_ext_ptr_t' + '\n')
    host_file.append('    // and provide the Banks' + '\n')
    host_file.append('    if (xcl::is_emulation()) {' + '\n')
    host_file.append('    	printf("Emulation Mode \\n");' + '\n')
    host_file.append('        for (int i = 0; i < NUM_KERNEL; i++) {' + '\n')
    host_file.append('            inputSearchSpaceBufExt[i].obj = searchspace_data_part[i].data();' + '\n')
    host_file.append('            inputSearchSpaceBufExt[i].param = 0;' + '\n')
    if (_memory_type == 'DDR4'):
        host_file.append('            inputSearchSpaceBufExt[i].flags = XCL_MEM_DDR_BANK1;' + '\n')    
    elif (_memory_type == 'HBM2'):
        host_file.append('            inputSearchSpaceBufExt[i].flags = bank[0];' + '\n')
    host_file.append('        }' + '\n')
    host_file.append('        outputResultDistBufExt.obj = hw_dist.data();' + '\n')
    host_file.append('        outputResultDistBufExt.param = 0;' + '\n')
    if (_memory_type == 'DDR4'):
        host_file.append('        outputResultDistBufExt.flags = XCL_MEM_DDR_BANK1; ' + '\n')    
    elif (_memory_type == 'HBM2'):
        host_file.append('        outputResultDistBufExt.flags = bank[0]; ' + '\n')
    host_file.append('        outputResultIdBufExt.obj = hw_id.data();' + '\n')
    host_file.append('        outputResultIdBufExt.param = 0;' + '\n')
    if (_memory_type == 'DDR4'):
        host_file.append('        outputResultIdBufExt.flags = XCL_MEM_DDR_BANK1;' + '\n')
    elif (_memory_type == 'HBM2'):
        host_file.append('        outputResultIdBufExt.flags = bank[0];' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    else{' + '\n')
    host_file.append('        for (int i = 0; i < NUM_KERNEL; i++) {' + '\n')
    host_file.append('            inputSearchSpaceBufExt[i].obj = searchspace_data_part[i].data();' + '\n')
    host_file.append('            inputSearchSpaceBufExt[i].param = 0;' + '\n')
    if (_memory_type == 'DDR4'):
        host_file.append('            inputSearchSpaceBufExt[i].flags = XCL_MEM_DDR_BANK1;' + '\n')    
    elif (_memory_type == 'HBM2'):
        host_file.append('            inputSearchSpaceBufExt[i].flags = bank[0];' + '\n')
    host_file.append('        }' + '\n')
    # slr + bank assignment matching
    num_pe_per_bank = int(math.ceil(float(_total_num_pe)/float(_num_mem_banks)))
    bank_idx = -1
    bank_cnt = 0
    pe_bank_location = []
    for pe_idx in range (_total_num_pe):
        bank_cnt = bank_cnt % num_pe_per_bank
        if (bank_cnt == 0):
            bank_idx += 1
        if (_memory_type == 'DDR4'):
            host_file.append('        inputSearchSpaceBufExt[' + str(pe_idx) + '].flags = XCL_MEM_DDR_BANK' + str(bank_idx) + ';' + '\n')
        elif (_memory_type == 'HBM2'):
            host_file.append('        inputSearchSpaceBufExt[' + str(pe_idx) + '].flags = bank[' + str(bank_idx) + '];' + '\n')
        pe_bank_location.append(bank_idx)
        bank_cnt += 1
    host_file.append('' + '\n')
    host_file.append('        outputResultDistBufExt.obj = hw_dist.data();' + '\n')
    host_file.append('        outputResultDistBufExt.param = 0;' + '\n')
    if (_memory_type == 'DDR4'):
        host_file.append('        outputResultDistBufExt.flags = XCL_MEM_DDR_BANK1; ' + '\n')
    elif (_memory_type == 'HBM2'):
        host_file.append('        outputResultDistBufExt.flags = bank[0]; ' + '\n')
    host_file.append('        outputResultIdBufExt.obj = hw_id.data();' + '\n')
    host_file.append('        outputResultIdBufExt.param = 0;' + '\n')
    if (_memory_type == 'DDR4'):
        host_file.append('        outputResultIdBufExt.flags = XCL_MEM_DDR_BANK1;' + '\n')
    elif (_memory_type == 'HBM2'):
        host_file.append('        outputResultIdBufExt.flags = bank[0];' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // These commands will allocate memory on the FPGA. The cl::Buffer objects can' + '\n')
    host_file.append('    // be used to reference the memory locations on the device.' + '\n')
    host_file.append('    //Creating Buffers' + '\n')
    host_file.append('    for (int i = 0; i < NUM_KERNEL; i++) {' + '\n')
    host_file.append('        OCL_CHECK(err,' + '\n')
    host_file.append('                  buffer_input_searchspace[i] =' + '\n')
    host_file.append('                      cl::Buffer(context,' + '\n')
    host_file.append('                                 CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,' + '\n')
    host_file.append('                                 sizeof(float) * (QUERY_FEATURE_RESERVE + part_size), ' + '\n')
    host_file.append('                                 &inputSearchSpaceBufExt[i],' + '\n')
    host_file.append('                                 &err));' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    OCL_CHECK(err,' + '\n')
    host_file.append('            buffer_output_dist_result =' + '\n')
    host_file.append('                cl::Buffer(context,' + '\n')
    host_file.append('                            CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,' + '\n')
    host_file.append('                            sizeof(float) * TOP,' + '\n')
    host_file.append('                            &outputResultDistBufExt,' + '\n')
    host_file.append('                            &err));' + '\n')
    host_file.append('    OCL_CHECK(err,' + '\n')
    host_file.append('            buffer_output_id_result =' + '\n')
    host_file.append('                cl::Buffer(context,' + '\n')
    host_file.append('                            CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,' + '\n')
    host_file.append('                            sizeof(int) * TOP,' + '\n')
    host_file.append('                            &outputResultIdBufExt,' + '\n')
    host_file.append('                            &err));                            ' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Copy input data to Device Global Memory' + '\n')
    host_file.append('    for (int i = 0; i < NUM_KERNEL; i++) {' + '\n')
    host_file.append('        OCL_CHECK(err,' + '\n')
    host_file.append('                  err = q.enqueueMigrateMemObjects(' + '\n')
    host_file.append('                      {buffer_input_searchspace[i]}, 0));' + '\n')
    host_file.append('    }' + '\n')
    host_file.append('    q.finish();' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Start timer' + '\n')
    host_file.append('    double kernel_time_in_sec = 0, result = 0;' + '\n')
    host_file.append('    std::chrono::duration<double> kernel_time(0);' + '\n')
    host_file.append('    auto kernel_start = std::chrono::high_resolution_clock::now();' + '\n')
    host_file.append('' + '\n')
    host_file.append('    int i = 0;' + '\n')
    host_file.append('    int count = 0;' + '\n')
    for i in range(len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            host_file.append('    int NUM_KERNEL_SLR' + str(i) + ' = ' + str(_slr_num_pe[i]) + ';' + '\n')
    for i in range(len(_slr_num_pe)):
        if (_slr_num_pe[i] > 0):
            host_file.append('    for (i = 0; i < NUM_KERNEL_SLR' + str(i) + '; i++) {' + '\n')
            host_file.append('        //Setting the compute kernel arguments' + '\n')
            host_file.append('        OCL_CHECK(err, err = cmpt_krnl_SLR' + str(i) + '.setArg(i*2+0, buffer_input_searchspace[count+i])); ' + '\n')
            host_file.append('        OCL_CHECK(err, err = cmpt_krnl_SLR' + str(i) + '.setArg(i*2+1, (count+i)*part_size/INPUT_DIM)); ' + '\n')
            host_file.append('    }' + '\n')
            host_file.append('    //Invoking the compute kernels' + '\n')
            host_file.append('    OCL_CHECK(err, err = q.enqueueTask(cmpt_krnl_SLR' + str(i) + '));' + '\n')
            host_file.append('    count += i;' + '\n')
    host_file.append('' + '\n')
    host_file.append('    //Setting the aggregate kernel arguments' + '\n')
    host_file.append('    int arg_idx = NUM_USED_SLR;' + '\n')
    host_file.append('    OCL_CHECK(err, err = aggregate_krnl.setArg(arg_idx, buffer_output_dist_result));' + '\n')
    host_file.append('    OCL_CHECK(err, err = aggregate_krnl.setArg(arg_idx+1, buffer_output_id_result));' + '\n')
    host_file.append('    //Invoking the aggregate kernel' + '\n')
    host_file.append('    OCL_CHECK(err, err = q.enqueueTask(aggregate_krnl));' + '\n')
    host_file.append('' + '\n')
    host_file.append('    q.finish();' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Stop timer' + '\n')
    host_file.append('    auto kernel_end = std::chrono::high_resolution_clock::now();' + '\n')
    host_file.append('    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);' + '\n')
    host_file.append('    kernel_time_in_sec = kernel_time.count();' + '\n')
    host_file.append('    std::cout << "Execution time = " << kernel_time_in_sec << std::endl;' + '\n')
    host_file.append('' + '\n')
    host_file.append('    // Copy Result from Device Global Memory to Host Local Memory' + '\n')
    host_file.append('    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(' + '\n')
    host_file.append('                   {buffer_output_dist_result, buffer_output_id_result}, ' + '\n')
    host_file.append('                   CL_MIGRATE_MEM_OBJECT_HOST));' + '\n')
    host_file.append('    q.finish();' + '\n')
    host_file.append('' + '\n')
    host_file.append('    bool match = true;' + '\n')
    host_file.append('    match = verify(sw_dist, hw_dist, sw_id, hw_id, TOP);' + '\n')
    host_file.append('    //OPENCL HOST CODE AREA ENDS' + '\n')
    host_file.append('' + '\n')
    host_file.append('    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;' + '\n')
    host_file.append('    return (match ? EXIT_SUCCESS : EXIT_FAILURE);' + '\n')
    host_file.append('}' + '\n')
 
    with open(host_file_name, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(host_file)
    return pe_bank_location
    
def Generate_Connectivity_Map(_memory_type, _pe_bank_location, _slr_num_pe):
    connectivity_file_name = 'knn.ini'
    connectivity_file = []
    overall_pe_idx = 0

    connectivity_file.append('[connectivity]' + '\n\n')
    for slr_idx in range (len(_slr_num_pe)):
        if (_slr_num_pe[slr_idx] > 0):
            connectivity_file.append('slr=krnl_partialKnn_SLR' + str(slr_idx) + '_1:SLR' + str(slr_idx) + '\n')
            for pe_idx in range (_slr_num_pe[slr_idx]):
                if (_memory_type == 'DDR4'):
                    connectivity_file.append('sp=krnl_partialKnn_SLR' + str(slr_idx) + '_1.searchSpace_' + str(pe_idx) + ':DDR[' + str(_pe_bank_location[overall_pe_idx]) + ']' + '\n')
                elif (_memory_type == 'HBM2'):
                    connectivity_file.append('sp=krnl_partialKnn_SLR' + str(slr_idx) + '_1.searchSpace_' + str(pe_idx) + ':HBM[' + str(_pe_bank_location[overall_pe_idx]) + ']' + '\n')
                overall_pe_idx += 1
    connectivity_file.append('' + '\n')

    input_idx = 0
    for slr_idx in range (len(_slr_num_pe)):
        if (_slr_num_pe[slr_idx] > 0):
            connectivity_file.append('stream_connect=krnl_partialKnn_SLR' + str(slr_idx) + '_1.out:krnl_globalSort_1.in' + str(input_idx) + '\n')
            input_idx += 1
    
    connectivity_file.append('' + '\n')

    connectivity_file.append('slr=krnl_globalSort_1:SLR1' + '\n')
    if (_memory_type == 'DDR4'):
        connectivity_file.append('sp=krnl_globalSort_1.output_knnDist:DDR[1]' + '\n')
        connectivity_file.append('sp=krnl_globalSort_1.output_knnId:DDR[1]' + '\n')
    elif (_memory_type == 'HBM2'):
        connectivity_file.append('sp=krnl_globalSort_1.output_knnDist:HBM[0]' + '\n')
        connectivity_file.append('sp=krnl_globalSort_1.output_knnId:HBM[0]' + '\n')
    connectivity_file.append('' + '\n')

    for slr_idx in range (len(_slr_num_pe)):
        if (_slr_num_pe[slr_idx] > 0):
            connectivity_file.append('nk=krnl_partialKnn_SLR' + str(slr_idx) + ':1' + '\n')
    connectivity_file.append('nk=krnl_globalSort:1' + '\n')

    with open(connectivity_file_name, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(connectivity_file)

def Generate_MakeFile(_num_slr, _kernel_freq):
    make_file_name = 'Makefile'
    make_file = []

    make_file.append('.PHONY: help' + '\n\n')
    make_file.append('help::' + '\n')
    make_file.append('	$(ECHO) "Makefile Usage:"' + '\n')
    make_file.append('	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"' + '\n')
    make_file.append('	$(ECHO) "      Command to generate the design for specified Target and Shell."' + '\n')
    make_file.append('	$(ECHO) ""' + '\n')
    make_file.append('	$(ECHO) "  make clean "' + '\n')
    make_file.append('	$(ECHO) "      Command to remove the generated non-hardware files."' + '\n')
    make_file.append('	$(ECHO) ""' + '\n')
    make_file.append('	$(ECHO) "  make cleanall"' + '\n')
    make_file.append('	$(ECHO) "      Command to remove all the generated files."' + '\n')
    make_file.append('	$(ECHO) ""' + '\n')
    make_file.append('	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"' + '\n')
    make_file.append('	$(ECHO) "      Command to run application in emulation."' + '\n')
    make_file.append('	$(ECHO) ""' + '\n')
    make_file.append('	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"' + '\n')
    make_file.append('	$(ECHO) "      Command to build xclbin application."' + '\n')
    make_file.append('	$(ECHO) ""' + '\n\n')
    make_file.append('# Points to top directory of Git repository' + '\n')
    make_file.append('COMMON_REPO = ./' + '\n')
    make_file.append('PWD = $(shell readlink -f .)' + '\n')
    make_file.append('ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))' + '\n\n')
    make_file.append('TARGET := hw' + '\n')
    make_file.append('HOST_ARCH := x86' + '\n\n')
    make_file.append('include ./utils.mk' + '\n\n')
    make_file.append('XSA := $(call device2xsa, $(DEVICE))' + '\n')
    make_file.append('TEMP_DIR := ./_x.$(TARGET).$(XSA)' + '\n')
    make_file.append('BUILD_DIR := ./build_dir.$(TARGET).$(XSA)' + '\n\n')
    make_file.append('VPP := v++' + '\n\n')
    make_file.append('#Include Libraries' + '\n')
    make_file.append('include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk' + '\n')
    make_file.append('include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk' + '\n')
    make_file.append('CXXFLAGS += $(xcl2_CXXFLAGS)' + '\n')
    make_file.append('LDFLAGS += $(xcl2_LDFLAGS)' + '\n')
    make_file.append('HOST_SRCS += $(xcl2_SRCS)' + '\n')
    make_file.append('CXXFLAGS += -pthread' + '\n')
    make_file.append('CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11' + '\n')
    make_file.append('LDFLAGS += $(opencl_LDFLAGS)' + '\n\n')
    make_file.append('HOST_SRCS += src/host.cpp ' + '\n')
    make_file.append('# Host compiler global settings' + '\n')
    make_file.append('CXXFLAGS += -fmessage-length=0' + '\n')
    make_file.append('LDFLAGS += -lrt -lstdc++ ' + '\n\n')
    make_file.append('# Kernel compiler global settings' + '\n')
    make_file.append('CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps --kernel_frequency ' + str(_kernel_freq) + '' + '\n')
    make_file.append('ifneq ($(TARGET), hw)' + '\n')
    make_file.append('	CLFLAGS += -g' + '\n')
    make_file.append('endif' + '\n\n')
    make_file.append('# Kernel linker flags' + '\n')
    make_file.append('LDCLFLAGS += --config src/knn.ini' + '\n\n')
    make_file.append('EXECUTABLE = knn' + '\n')
    make_file.append('CMD_ARGS = $(BUILD_DIR)/knn.xclbin' + '\n')
    make_file.append('EMCONFIG_DIR = $(TEMP_DIR)' + '\n\n')
    make_file.append('BINARY_CONTAINERS += $(BUILD_DIR)/knn.xclbin' + '\n')
    for i in range (_num_slr):
        make_file.append('BINARY_CONTAINER_knn_OBJS += $(TEMP_DIR)/krnl_partialKnn_SLR' + str(i) + '.xo' + '\n')
    make_file.append('BINARY_CONTAINER_knn_OBJS += $(TEMP_DIR)/krnl_globalSort.xo' + '\n')
    make_file.append('CP = cp -rf' + '\n\n')
    make_file.append('.PHONY: all clean cleanall docs emconfig' + '\n')
    make_file.append('all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig' + '\n\n')
    make_file.append('.PHONY: exe' + '\n')
    make_file.append('exe: $(EXECUTABLE)' + '\n\n')
    make_file.append('.PHONY: build' + '\n')
    make_file.append('build: $(BINARY_CONTAINERS)' + '\n\n')

    for i in range (_num_slr):
        make_file.append('# Building kernel' + '\n')
        make_file.append('$(TEMP_DIR)/krnl_partialKnn_SLR' + str(i) + '.xo: src/krnl_partialKnn_SLR' + str(i) + '.cpp' + '\n')
        make_file.append('	mkdir -p $(TEMP_DIR)' + '\n')
        make_file.append('	$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k krnl_partialKnn_SLR' + str(i) + ' -I\'$(<D)\' -o\'$@\' \'$<\'' + '\n')
    make_file.append('$(TEMP_DIR)/krnl_globalSort.xo: src/krnl_globalSort.cpp' + '\n')
    make_file.append('	mkdir -p $(TEMP_DIR)' + '\n')
    make_file.append('	$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k krnl_globalSort -I\'$(<D)\' -o\'$@\' \'$<\'' + '\n')
    make_file.append('$(BUILD_DIR)/knn.xclbin: $(BINARY_CONTAINER_knn_OBJS)' + '\n')
    make_file.append('	mkdir -p $(BUILD_DIR)' + '\n')
    make_file.append('	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o\'$@\' $(+)' + '\n\n')

    make_file.append('# Building Host' + '\n')
    make_file.append('$(EXECUTABLE): check-xrt $(HOST_SRCS) $(HOST_HDRS)' + '\n')
    make_file.append('	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o \'$@\' $(LDFLAGS)' + '\n\n')
    make_file.append('emconfig:$(EMCONFIG_DIR)/emconfig.json' + '\n')
    make_file.append('$(EMCONFIG_DIR)/emconfig.json:' + '\n')
    make_file.append('	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)' + '\n\n')
    make_file.append('check: all' + '\n')
    make_file.append('ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))' + '\n')
    make_file.append('	$(CP) $(EMCONFIG_DIR)/emconfig.json .' + '\n')
    make_file.append('	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(BUILD_DIR)/knn.xclbin' + '\n')
    make_file.append('else' + '\n')
    make_file.append('	./$(EXECUTABLE) $(BUILD_DIR)/knn.xclbin' + '\n')
    make_file.append('endif' + '\n\n')
    make_file.append('# Cleaning stuff' + '\n')
    make_file.append('clean:' + '\n')
    make_file.append('	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} ' + '\n')
    make_file.append('	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv ' + '\n')
    make_file.append('	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb' + '\n\n')
    make_file.append('cleanall: clean' + '\n')
    make_file.append('	-$(RMDIR) build_dir* sd_card*' + '\n')
    make_file.append('	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin' + '\n\n')

    with open(make_file_name, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(make_file)