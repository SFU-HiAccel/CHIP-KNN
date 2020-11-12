#!/usr/bin/python 

import os
import re
import math 
from bandwidth_interpolation import *

# Generate_SinglePE_Design(_N, _D, _Dist, _K, _port_width, _buf_size)
# Funtion usage: "Generate_SinglePE_Design(4194304, 2, 0, 10, 512, 131072)"
'''
# _N: number of search space data points 
# _D: number of query data dimension
# _Dist: type of distance metric: 0=Manhattan 1=Euclidean
# _K: number of nearest neighbors to return 
# _port_width: data port width 
# _buf_size: on-chip BRAM buffer size 
# _memory_type: DDR4 or HBM2
'''
def Generate_SinglePE_Design(_N, _D, _Dist, _K, _port_width, _buf_size, _memory_type):
    parallel_sort = Generate_Design_Configuration(_N, _D, _Dist, _K, _port_width, _buf_size, _memory_type)
    Generate_Design_Core(parallel_sort, 1, 'krnl_partialKnn', 'SLR0') # parallel_sort, num_PE, kernel_name, SLR_name)

def Generate_Design_Configuration(_N, _D, _Dist, _K, _port_width, _buf_size, _memory_type):
    config_file = 'krnl_config.h'

    INPUT_DIM = _D
    TOP = _K
    NUM_SP_PTS = _N
    DISTANCE_METRIC = _Dist

    NUM_KERNEL = 1
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

def Generate_Design_Core(_parallel_sort, _num_PE, _krnl_top_name, _SLR_name):
    core_krnl_file_name = _krnl_top_name + '.cpp'
    core_krnl_file = []

    GenerateHeader(core_krnl_file, _num_PE)
    GenerateLoad(core_krnl_file, _num_PE, _SLR_name)
    GenerateCompute(core_krnl_file, _parallel_sort, _SLR_name)
    GenerateSort(core_krnl_file, _parallel_sort, _SLR_name)
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
    _core_krnl_file.append('    hls::stream<pkt> &out1)' + '\n')
    _core_krnl_file.append('{' + '\n')
    for idx in range (_num_PE):
        _core_krnl_file.append('#pragma HLS INTERFACE m_axi port=searchSpace_' + str(idx) + ' offset=slave bundle=gmem0' + '\n')
        _core_krnl_file.append('#pragma HLS INTERFACE s_axilite port=searchSpace_' + str(idx) + ' bundle=control' + '\n')
        _core_krnl_file.append('#pragma HLS INTERFACE s_axilite port=start_id_' + str(idx) + ' bundle=control' + '\n')
    _core_krnl_file.append('#pragma HLS INTERFACE axis port=out1' + '\n')
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
            _core_krnl_file.append('				sort_SLR0(sort_flag, start_id_' + str(idx) + '+(i-2)*DIS_LEN, local_distance_' + str(idx) + '_A, local_kNearstDist_partial_' + str(idx) + ', local_kNearstId_partial_' + str(idx) + ');' + '\n')
    elif (_parallel_sort == 0):
        for idx in range (_num_PE):
            _core_krnl_file.append('				sort_SLR0(sort_flag, start_id_' + str(idx) + '+(i-2)*DIS_LEN, local_distance_' + str(idx) + '_A, local_kNearstDist[' + str(idx) + '], local_kNearstId[' + str(idx) + ']);' + '\n')
    
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

    _core_krnl_file.append('        merge_global_' + _SLR_name + '(local_kNearstDist, local_kNearstId, global_kNearstDist, global_kNearstId);' + '\n')

    _core_krnl_file.append('	}' + '\n\n')
    _core_krnl_file.append('    STREAM_WIDTH v_data;' + '\n')
    _core_krnl_file.append('    float temp_data;' + '\n')
    _core_krnl_file.append('	DIST_OUT: for (int i = 1; i < TOP+1; ++i){' + '\n')
    _core_krnl_file.append('	#pragma HLS PIPELINE II=1' + '\n')
    _core_krnl_file.append('        temp_data = global_kNearstDist[i]; ' + '\n')
    _core_krnl_file.append('        v_data.range(31, 0) = *((uint32_t *)(&temp_data));' + '\n')
    _core_krnl_file.append('        pkt v;' + '\n')
    _core_krnl_file.append('		v.data = v_data;' + '\n')
    _core_krnl_file.append('		out1.write(v);' + '\n')
    _core_krnl_file.append('	}' + '\n')
    _core_krnl_file.append('	ID_OUT: for (int i = 1; i < TOP+1; ++i){' + '\n')
    _core_krnl_file.append('	#pragma HLS PIPELINE II=1' + '\n')
    _core_krnl_file.append('		pkt v_id;' + '\n')
    _core_krnl_file.append('		v_id.data = global_kNearstId[i]; ' + '\n')
    _core_krnl_file.append('		out1.write(v_id);' + '\n')
    _core_krnl_file.append('	}' + '\n')
    _core_krnl_file.append('	return;' + '\n')
    _core_krnl_file.append('}' + '\n')
    _core_krnl_file.append('}' + '\n')