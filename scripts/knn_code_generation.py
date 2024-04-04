#!/usr/bin/python 

import os
import re
import math 
import sys
from config import *
import bandwidth_interpolation as bw

"""
Each function in this script is meant to generate a part of the KNN HLS C++ code. 
The common inputs are as follows:
    _core_krnl_file:    The name of the CPP file to be generated.
    _knn_config:        A configuration object, containing all of the configuration variables that were generated.
"""

class KnnConfigClass:
    def __init__(self, _num_PE, _using_segments, _sort_II, _data_type_total_sz, _local_dist_sz, _iwidth,
                    _D2L_factor_w, _D2I_factor_w, _I2D_factor_w, _using_Ltypes, _num_segments, _input_dim, 
                    _using_float, _using_fixedpt, _approx_num_hiermerge_nodes,
                    _approx_num_hiermerge_stages,
                    _using_intra_pe_merge):
        self.num_PE                     = _num_PE
        self.using_segments             = _using_segments
        self.sort_II                    = _sort_II
        self.data_type_total_sz         = _data_type_total_sz
        self.local_dist_sz              = _local_dist_sz
        self.iwidth                     = _iwidth
        self.D2L_factor_w               = _D2L_factor_w
        self.D2I_factor_w               = _D2I_factor_w
        self.I2D_factor_w               = _I2D_factor_w
        self.using_Ltypes               = _using_Ltypes
        self.num_segments               = _num_segments
        self.input_dim                  = _input_dim
        self.using_float                = _using_float
        self.using_fixedpt              = _using_fixedpt
        self.approx_num_hiermerge_nodes = _approx_num_hiermerge_nodes
        self.approx_num_hiermerge_stages = _approx_num_hiermerge_stages
        self.using_intra_pe_merge   = _using_intra_pe_merge

def Generate_Design_Configuration(_N, _D, _Dist, _K, _port_width, _memory_type, data_type_to_use, data_type_int_bits, data_type_fract_bits, _total_num_pe):
    config_file = 'knn.h'

    INPUT_DIM = _D
    TOP = _K
    NUM_SP_PTS = _N
    DISTANCE_METRIC = _Dist

    NUM_PE = _total_num_pe

    IWIDTH = _port_width
    legal_total_bits = [8, 16, 32]

    DATA_TYPE_TOTAL_SZ = data_type_int_bits + data_type_fract_bits
    if (data_type_to_use == "int"):
        data_type_to_use = "signed fixed"
        data_type_int_bits = 32
        data_type_fract_bits = 0

    print("Data type to use = " + str(data_type_to_use))
    if (data_type_to_use == "float"):
        DATA_TYPE_TOTAL_SZ = 32
    elif (data_type_to_use == "signed fixed"):
        pass
    elif (data_type_to_use == "unsigned fixed"):
        print("\n\n\n\n\n\n CAREFUL! Unsigned fixed is not fully tested yet as of Sept 30 2021.\n\n\n\n\n")
    else:
        print("Please specify a valid data type!")
        print(data_type_to_use)
        sys.exit(-1)

    if (data_type_to_use == "float"):
        USING_FLOAT = 1
        USING_FIXEDPT = 0
    else: 
        USING_FLOAT = 0
        USING_FIXEDPT = 1

    MAX_FIXEDPT_VAL             = 2**(data_type_int_bits - 1) - 2**(-1*data_type_fract_bits)
    FLOOR_SQRT_MAX_FIXEDPT_VAL  = int(math.sqrt(int(MAX_FIXEDPT_VAL)))
    MAX_FLT_VAL                 = "3.402823e+38f"
    FLOOR_SQRT_MAX_FLT_VAL      = "1.8446742e+19f"


    if ("fixed" in data_type_to_use):
        if DATA_TYPE_TOTAL_SZ not in legal_total_bits:
            print("For fixed-point, please specify a valid number of bits!")
            sys.exit(-1)

    SORT_II = 2

    new_file = []
    baseDirName = os.getcwd()

    #########################################################
    ### Begin writing stuff to the file.
    #########################################################

    new_file.append("#include \"ap_int.h\" " + "\n")
    new_file.append("#include \"ap_axi_sdata.h\" " + "\n")
    new_file.append("#include <tapa.h>" + "\n")
    new_file.append("#include <inttypes.h>" + "\n")
    new_file.append("#include <stdlib.h>" + "\n")
    new_file.append("" + "\n")
    new_file.append("" + "\n")
    new_file.append("// CEIL_DIVISION(X, Y) = ceil(x/y)." + '\n')
    new_file.append("#define CEIL_DIVISION(X, Y) ( (X-1)/Y + 1 )" + '\n')
    new_file.append("// ROUND_TO_NEXT_MULTIPLE(X, Y) rounds X up to the nearest multiple of Y." + '\n')
    new_file.append("#define ROUND_TO_NEXT_MULTIPLE(X, Y) (CEIL_DIVISION(X,Y) * Y)" + '\n')
    new_file.append("" + "\n")
    new_file.append("" + "\n")
    new_file.append("const int IWIDTH = " + str(IWIDTH) + ";" + "\n")
    new_file.append("#define INTERFACE_WIDTH ap_uint<IWIDTH> " + "\n")
    new_file.append("const int  INPUT_DIM = "             + str(INPUT_DIM) + ";" + "\n")
    new_file.append("const int  TOP = "                   + str(TOP) + ";" + "\n")
    new_file.append("#define NUM_SP_PTS "           + "(" + str(NUM_SP_PTS) + ")" + "\n")
    new_file.append("#define DISTANCE_METRIC "      + "(" + str(DISTANCE_METRIC) + ")" + "\n")
    new_file.append("#define NUM_PE "               + "(" + str(NUM_PE) + ")" + "\n")
    new_file.append("" + "\n")

    if (USING_FLOAT):
        LOCAL_DIST_SZ = 32
        new_file.append("#define DATA_TYPE_TOTAL_SZ 32" + "\n")
        new_file.append("#define DATA_TYPE float" + "\n")
        new_file.append("#define LOCAL_DIST_SZ   (32)" + "\n")
        new_file.append("#define LOCAL_DIST_DTYPE float" + "\n")
        new_file.append("#define TRANSFER_TYPE ap_uint<DATA_TYPE_TOTAL_SZ>" + "\n")

    elif (USING_FIXEDPT):
        # We need to have L2I_FACTOR_W as close to D2I_FACTOR_W as possible, while maintaining L2I >= D2L.
        # Thus, LOCAL_DIST_SZ must be the largest power of two that is smaller than or equal to IWIDTH * DATA_TYPE_TOTAL_SZ.
        initial_LSize_guess = int(math.floor(math.sqrt(DATA_TYPE_TOTAL_SZ * IWIDTH)))   # Geomean of DType size and IType size
        LOCAL_DIST_SZ = 2 ** int(math.floor(math.log2(initial_LSize_guess)))            # Floor the geomean to a power of 2
        D2L_FACTOR_W = math.ceil( LOCAL_DIST_SZ / (DATA_TYPE_TOTAL_SZ) )
        D2I_FACTOR_W = math.ceil( IWIDTH / (INPUT_DIM * DATA_TYPE_TOTAL_SZ) )

        if (D2I_FACTOR_W <= 1):
            ### In this case, compute is our bottleneck. Don't parallelize sort.
            LOCAL_DIST_SZ = DATA_TYPE_TOTAL_SZ
        else:
            while (D2L_FACTOR_W > D2I_FACTOR_W):
                LOCAL_DIST_SZ = int(LOCAL_DIST_SZ/2)
                D2L_FACTOR_W = math.ceil( LOCAL_DIST_SZ / (DATA_TYPE_TOTAL_SZ) )

        new_file.append("#define DATA_TYPE_INT_PART_SZ " + str(int(data_type_int_bits)) +" // NOTE: This includes the sign bit, if applicable." + "\n")
        new_file.append("#define DATA_TYPE_TOTAL_SZ    " + str(int(DATA_TYPE_TOTAL_SZ)) + "\n")
        new_file.append("#define DATA_TYPE ap_" + ("u" if data_type_to_use == "unsigned fixed" else "") + "fixed<DATA_TYPE_TOTAL_SZ, DATA_TYPE_INT_PART_SZ, \\" + "\n")
        new_file.append("                                AP_RND, AP_SAT>" + "\n")
        new_file.append("" + "\n")

        new_file.append("// Datapacking compute's output & sort's input" + "\n")
        new_file.append("#define LOCAL_DIST_SZ   (" + str(int(LOCAL_DIST_SZ)) + ")" + "\n")
        new_file.append("#define LOCAL_DIST_DTYPE ap_uint<LOCAL_DIST_SZ>" + "\n")

        #if (DATA_TYPE_TOTAL_SZ < 32):
        #    new_file.append("#define INT32 ap_uint<32>" + "\n")

    new_file.append("#define INT32 ap_uint<32>" + "\n")
    new_file.append("" + "\n")
    new_file.append("" + "\n")

    new_file.append("/***************************************************************/" + "\n")
    new_file.append("" + "\n")


    # Dtypes are the 'basic' data type; that is, each point in the search space has coordinates of type DATA_TYPE.
    # Ltypes are Dtypes, after they've been data-packed into the 'local_distance' arrays.
    # Itypes are the interface width, and specify how wide the port is from off-chip DRAM, to the FPGA chip.
    # So, in terms of bit-widths, D-type bitwidth <= L-type bitwidth <= I-type bitwidth 
    L2I_FACTOR_W = math.ceil( IWIDTH / (INPUT_DIM * LOCAL_DIST_SZ) )
    D2L_FACTOR_W = math.ceil( LOCAL_DIST_SZ / (DATA_TYPE_TOTAL_SZ) )
    D2I_FACTOR_W = math.ceil( IWIDTH / (INPUT_DIM * DATA_TYPE_TOTAL_SZ) )
    I2D_FACTOR_W = math.ceil( (INPUT_DIM * DATA_TYPE_TOTAL_SZ) / IWIDTH )

    L2I_FACTOR_W_fraction = IWIDTH / (INPUT_DIM * LOCAL_DIST_SZ)
    D2I_FACTOR_W_fraction = IWIDTH / (INPUT_DIM * DATA_TYPE_TOTAL_SZ) 

    new_file.append("// L2I = Local to Interface" + '\n')
    new_file.append("const int L2I_FACTOR_W = CEIL_DIVISION( IWIDTH, (INPUT_DIM * LOCAL_DIST_SZ) );" + '\n')
    new_file.append("// D2L = Data_Type to Local" + '\n')
    new_file.append("const int D2L_FACTOR_W = CEIL_DIVISION(LOCAL_DIST_SZ , DATA_TYPE_TOTAL_SZ);" + '\n')
    new_file.append("// D2I = Data_Type to Interface" + '\n')
    new_file.append("const int D2I_FACTOR_W = CEIL_DIVISION(IWIDTH, (INPUT_DIM * DATA_TYPE_TOTAL_SZ));" + '\n')
    new_file.append("// I2D = Interface to Data_type" + '\n')
    new_file.append("const int I2D_FACTOR_W = CEIL_DIVISION( (INPUT_DIM * DATA_TYPE_TOTAL_SZ), IWIDTH);" + '\n')


    ########################################
    ### USING_LTYPES:
    ########################################
    ## If there are multiple D-values from a single I value, we make use of the intermediate data type (L-type)
    if (D2I_FACTOR_W_fraction > 1 / SORT_II):
        USING_LTYPES = 1

    elif (D2I_FACTOR_W_fraction <= 1 / SORT_II):
        USING_LTYPES = 0

    new_file.append('#define USING_LTYPES ' + str(int(USING_LTYPES)) + "\n")

    ########################################
    ### USING_SEGMENTS and NUM_SEGMENTS:
    ########################################
    ## If there are multiple L-values calculated from a single I-value, we need parallel sorting
    ## to perform load-balancing.
    if ( L2I_FACTOR_W_fraction > 1.0 / SORT_II ):
        USING_SEGMENTS = 1
        NUM_SEGMENTS = L2I_FACTOR_W * SORT_II
        new_file.append("#define USING_SEGMENTS " + "(" + str(USING_SEGMENTS) + ")" + "\n")
        new_file.append("#define NUM_SEGMENTS (L2I_FACTOR_W * " + str(SORT_II) + ")" + "\n")

    elif ( L2I_FACTOR_W_fraction <= 1.0 / SORT_II ):
        USING_SEGMENTS = 0
        NUM_SEGMENTS = 1
        new_file.append("#define USING_SEGMENTS " + "(" + str(USING_SEGMENTS) + ")" + "\n")
        new_file.append("#define NUM_SEGMENTS (1)" + "\n")


    new_file.append("" + "\n")
    new_file.append("// Round up to the nearest multiple, because otherwise some logic breaks (incorrect sizes => bad logic in edgecases)" + "\n")
    new_file.append("#define NUM_SP_PTS_PER_KRNL_PADDED ROUND_TO_NEXT_MULTIPLE(CEIL_DIVISION(NUM_SP_PTS, NUM_PE), (NUM_SEGMENTS*D2I_FACTOR_W))" + '\n')
    new_file.append("#define NUM_BYTES_PER_KRNL_PADDED (NUM_SP_PTS_PER_KRNL_PADDED * DATA_TYPE_TOTAL_SZ/8 * INPUT_DIM)" + '\n')
    new_file.append("" + "\n")

    new_file.append("// We partition the input points, so each PE gets it's own partition, to maximize parallelization." + '\n')
    new_file.append("const int PARTITION_LEN_IN_I = (NUM_BYTES_PER_KRNL_PADDED / (IWIDTH/8));" + '\n')
    new_file.append("const int PARTITION_LEN_IN_D = (NUM_BYTES_PER_KRNL_PADDED / (INPUT_DIM * DATA_TYPE_TOTAL_SZ/8));" + '\n')
    new_file.append("const int PARTITION_LEN_IN_L = (NUM_BYTES_PER_KRNL_PADDED / (INPUT_DIM * LOCAL_DIST_SZ/8));" + '\n')


    ########################################
    ### USING_INTRA_PE_MERGE
    ########################################
    if (USING_SEGMENTS or USING_LTYPES):
        USING_INTRA_PE_MERGE = 1
    else:
        USING_INTRA_PE_MERGE = 0


    ########################################
    ### APPROX_NUM_HIERMERGE_NODES:
    ########################################
    if (USING_SEGMENTS or D2L_FACTOR_W > 1):
        ## The number of STAGES in our hiermerge = log2(D2L * NUM_SEGMENTS * NUM_PE).
        ## The number of NODES is 1 + 2 + 4 + ... (#STAGES terms).
        ## This is (2^STAGES)-1. Therefore, the number of nodes is (D2L*NUM_SEG*NUM_PE)-1.
        APPROX_NUM_HIERMERGE_NODES  = D2L_FACTOR_W*NUM_SEGMENTS*NUM_PE - 1
    else:
        if (NUM_PE == 1):
            APPROX_NUM_HIERMERGE_NODES = 0
        else:
            APPROX_NUM_HIERMERGE_NODES = NUM_PE
    
    if (APPROX_NUM_HIERMERGE_NODES > 0):
        APPROX_NUM_HIERMERGE_STAGES = int(math.log2(APPROX_NUM_HIERMERGE_NODES))
    else:
        APPROX_NUM_HIERMERGE_STAGES = 0 

    ########################################
    ### SEGMENT_SIZE_IN_(x):
    ########################################
    if (USING_SEGMENTS == 1):
        new_file.append("// We name each sub-array of the local_distance arrays a \"segment\"." + "\n")
        new_file.append("#define SEGMENT_SIZE_IN_I (PARTITION_LEN_IN_I / NUM_SEGMENTS)" + '\n')
        new_file.append("#define SEGMENT_SIZE_IN_L (PARTITION_LEN_IN_L / NUM_SEGMENTS)" + '\n')
        new_file.append("#define SEGMENT_SIZE_IN_D (PARTITION_LEN_IN_D / NUM_SEGMENTS)" + '\n')


    new_file.append("" + "\n")
    new_file.append("//const int SWIDTH = DATA_TYPE_TOTAL_SZ; " + "\n")
    new_file.append("//typedef ap_axiu<SWIDTH, 0, 0, 0> pkt; " + "\n")
    new_file.append("//typedef ap_axiu<32, 0, 0, 0>    id_pkt;" + "\n")
    new_file.append("//#define STREAM_WIDTH ap_uint<SWIDTH> " + "\n")
    new_file.append("" + "\n")


    QUERY_FEATURE_RESERVE = 128
    new_file.append("const int NUM_FEATURES_PER_READ = (IWIDTH/DATA_TYPE_TOTAL_SZ);" + "\n")
    new_file.append("const int QUERY_FEATURE_RESERVE = (" + str(QUERY_FEATURE_RESERVE) + ");" + "\n")
    new_file.append("#define QUERY_DATA_RESERVE (QUERY_FEATURE_RESERVE / NUM_FEATURES_PER_READ)" + "\n")


    if (USING_FLOAT):
        new_file.append("#define MAX_DATA_TYPE_VAL (" + str(MAX_FLT_VAL) + ")" + "\n")
        new_file.append("#define FLOOR_SQRT_MAX_DATA_TYPE_VAL (" + str(FLOOR_SQRT_MAX_FLT_VAL) + ")" + "\n")
        new_file.append("" + "\n")
    elif (USING_FIXEDPT):
        new_file.append("#define MAX_DATA_TYPE_VAL (" + str(MAX_FIXEDPT_VAL) + ")" + "\n")
        new_file.append("#define FLOOR_SQRT_MAX_DATA_TYPE_VAL (" + str(FLOOR_SQRT_MAX_FIXEDPT_VAL) + ")" + "\n")
        new_file.append("" + "\n")


    #NUM_ITERATIONS = 3          # 10000
    #new_file.append("const int NUM_ITERATIONS = " + str(NUM_ITERATIONS) + ";" + "\n")
 
    with open(config_file, 'w') as f:
        # go to start of file
        f.seek(0)
        # actually write the lines
        f.writelines(new_file)

    print("\n\n")
    print("CHIPKNN PYTHON: CONFIGURATION FINISHED. Parallel_sort = {}, D2L = {}, NUM_SEGMENTS = {}".format(USING_SEGMENTS, D2L_FACTOR_W, NUM_SEGMENTS))

    knn_config = KnnConfigClass(NUM_PE, USING_SEGMENTS, SORT_II, DATA_TYPE_TOTAL_SZ, LOCAL_DIST_SZ, IWIDTH,
                                D2L_FACTOR_W, D2I_FACTOR_W, I2D_FACTOR_W, USING_LTYPES, NUM_SEGMENTS, INPUT_DIM, 
                                USING_FLOAT, USING_FIXEDPT, APPROX_NUM_HIERMERGE_NODES,
                                APPROX_NUM_HIERMERGE_STAGES,
                                USING_INTRA_PE_MERGE)

    return knn_config


#############################################################
### HEADER & LOAD
#############################################################


def GeneratePartialKNN_Header(_core_krnl_file, _knn_config):
    _core_krnl_file.append('#include "knn.h"' + '\n')
    _core_krnl_file.append('static inline DATA_TYPE absval(DATA_TYPE input){' + '\n')
    _core_krnl_file.append('    return (input > 0 ? input : static_cast<DATA_TYPE>(-1*input));' + '\n')
    _core_krnl_file.append('}' + '\n\n')


    ########################################
    ### Generating macros for sort:
    
    if (_knn_config.using_segments):
        _core_krnl_file.append('#define SORT_TO_HIERMERGE_STREAM_ARGS(PE, SEG) \\' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('    sort_to_hiermerge_dist_stream_##PE[D2L_FACTOR_W*SEG + ' + str(d2l_idx) + '],  \\' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('    sort_to_hiermerge_id_stream_##PE[D2L_FACTOR_W*SEG + ' + str(d2l_idx) + ']')
            if (d2l_idx != _knn_config.D2L_factor_w - 1):
                _core_krnl_file.append(',  \\' + "\n")

    elif (_knn_config.using_Ltypes):
        _core_krnl_file.append('#define SORT_TO_HIERMERGE_STREAM_ARGS(PE) \\' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('    sort_to_hiermerge_dist_stream_##PE[' + str(d2l_idx) + '],  \\' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('    sort_to_hiermerge_id_stream_##PE[' + str(d2l_idx) + ']')
            if (d2l_idx != _knn_config.D2L_factor_w - 1):
                _core_krnl_file.append(',  \\' + "\n")

    elif (_knn_config.using_intra_pe_merge == 0):
        _core_krnl_file.append('#define SORT_TO_HIERMERGE_STREAM_ARGS(PE) \\' + "\n")
        _core_krnl_file.append('    L0_out_dist[PE],  \\' + "\n")
        _core_krnl_file.append('    L0_out_id[PE]' + "\n")

    else:
        print("FATAL ERROR: In defining SORT_TO_HIERMERGE_STREAM_ARGS.")
        print("using_intra_pe_merge = {}".format(_knn_config.using_intra_pe_merge))
        print("using_segments = {}".format(_knn_config.using_segments))
        print("using_Ltypes = {}".format(_knn_config.using_Ltypes))
        sys.exit(-5)


    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")


    # NOTE: This logic still works even if NUM_SEGMENTS == 1.
    _core_krnl_file.append('#define INVOKE_PPS_UNITS_FOR_PE(PE) \\' + "\n")
    if (_knn_config.using_segments == 0):
        _core_krnl_file.append('    .invoke(para_partial_sort, PE, compute_to_sort_stream_##PE, SORT_TO_HIERMERGE_STREAM_ARGS(PE) ) \\' + "\n")
    else:
        for segment_idx in range(_knn_config.num_segments):
            _core_krnl_file.append('    .invoke(para_partial_sort, PE, ' + str(segment_idx) + ', compute_to_sort_stream_##PE[' + str(segment_idx) + '], SORT_TO_HIERMERGE_STREAM_ARGS(PE, ' + str(segment_idx) + ') ) \\' + "\n")

    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")



    #################################
    ### Macros for hiermerge:
    #################################
    if (_knn_config.using_intra_pe_merge):
        total_num_hiermerge_layers = int(math.log2(_knn_config.num_segments * _knn_config.D2L_factor_w))

        if (total_num_hiermerge_layers == 1):
            ## If there is only one hiermerge layer, we need to merge directly from the sort_to_hiermerge streams to the global merge streams
            _core_krnl_file.append('// purposefully empty #define' + "\n")
            _core_krnl_file.append('#define HIERMERGE_STREAM_DECLS(PE) \\' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('' + "\n")

            _core_krnl_file.append('#define INVOKE_HIERMERGE_UNITS_FOR_PE(PE) \\' + "\n")
            _core_krnl_file.append('    .invoke(merge_dual_streams, PE, ' +
                    '0, ' +
                    str(total_num_hiermerge_layers) + ', ' +
                    'sort_to_hiermerge_dist_stream_##PE[0], ' +
                    'sort_to_hiermerge_id_stream_##PE  [0], ' +
                    'sort_to_hiermerge_dist_stream_##PE[1], ' +
                    'sort_to_hiermerge_id_stream_##PE  [1], ' +
                    'L0_out_dist[PE],'     +
                    'L0_out_id[PE])'      + "\n")

        else:
            _core_krnl_file.append('#define HIERMERGE_STREAM_DECLS(PE) \\' + "\n")
            for cur_hiermerge_layer in range(1, total_num_hiermerge_layers):
                divider = 2**cur_hiermerge_layer
                _core_krnl_file.append('    tapa::streams<DATA_TYPE,    NUM_SEGMENTS*D2L_FACTOR_W/' + str(divider) + '    , TOP>    PE_##PE##_dist_stage' + str(cur_hiermerge_layer) + '; \\' + "\n")
                _core_krnl_file.append('    tapa::streams<int,          NUM_SEGMENTS*D2L_FACTOR_W/' + str(divider) + '    , TOP>    PE_##PE##_id_stage' + str(cur_hiermerge_layer) + '; \\' + "\n")

            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('' + "\n")

            _core_krnl_file.append('#define INVOKE_HIERMERGE_UNITS_FOR_PE(PE) \\' + "\n")
            for cur_hiermerge_layer in range(1, total_num_hiermerge_layers):
                num_nodes_in_cur_layer = 2**( total_num_hiermerge_layers - (cur_hiermerge_layer-1) )
                for node_idx in range(0, num_nodes_in_cur_layer, 2):
                    if (cur_hiermerge_layer == 1):
                        _core_krnl_file.append('    .invoke(merge_dual_streams, PE, ' +
                                str(node_idx) + ', ' +
                                str(cur_hiermerge_layer) + ', ' +
                                'sort_to_hiermerge_dist_stream_##PE[' + str(node_idx) + '], ' +
                                'sort_to_hiermerge_id_stream_##PE  [' + str(node_idx) + '], ' +
                                'sort_to_hiermerge_dist_stream_##PE[' + str(node_idx+1) + '], ' +
                                'sort_to_hiermerge_id_stream_##PE  [' + str(node_idx+1) + '], ' +
                                'PE_##PE##_dist_stage'    + str(cur_hiermerge_layer) +    '[' + str(int(node_idx/2)) + '], ' +
                                'PE_##PE##_id_stage'      + str(cur_hiermerge_layer) +    '[' + str(int(node_idx/2)) + ']) \\' + "\n")
                    else:
                        _core_krnl_file.append('    .invoke(merge_dual_streams, PE, ' +
                                str(node_idx) + ', ' +
                                str(cur_hiermerge_layer) + ', ' +
                                'PE_##PE##_dist_stage'    + str(cur_hiermerge_layer-1) +  '[' + str(node_idx) + '], ' +
                                'PE_##PE##_id_stage'      + str(cur_hiermerge_layer-1) +  '[' + str(node_idx) + '], ' +
                                'PE_##PE##_dist_stage'    + str(cur_hiermerge_layer-1) +  '[' + str(node_idx+1) + '], ' +
                                'PE_##PE##_id_stage'      + str(cur_hiermerge_layer-1) +  '[' + str(node_idx+1) + '], ' +
                                'PE_##PE##_dist_stage'    + str(cur_hiermerge_layer) +    '[' + str(int(node_idx/2)) + '], ' +
                                'PE_##PE##_id_stage'      + str(cur_hiermerge_layer) +    '[' + str(int(node_idx/2)) + ']) \\' + "\n")

            _core_krnl_file.append('    .invoke(merge_dual_streams, PE, ' +
                    '0, ' +
                    str(total_num_hiermerge_layers) + ', ' +
                    'PE_##PE##_dist_stage'    + str(total_num_hiermerge_layers-1) +  '[0], ' +
                    'PE_##PE##_id_stage'      + str(total_num_hiermerge_layers-1) +  '[0], ' +
                    'PE_##PE##_dist_stage'    + str(total_num_hiermerge_layers-1) +  '[1], ' +
                    'PE_##PE##_id_stage'      + str(total_num_hiermerge_layers-1) +  '[1], ' +
                    'L0_out_dist[PE],'     +
                    'L0_out_id[PE])'      + "\n")

        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
    ########################################



def GeneratePartialKNN_Load(_core_krnl_file, _knn_config):
    _core_krnl_file.append('/*************************************************/' + "\n")
    _core_krnl_file.append('/******************** LOADS: *********************/' + "\n")
    _core_krnl_file.append('/*************************************************/' + "\n")

    _core_krnl_file.append('void load_KNN(   int debug_PE_ID,' + "\n")
    _core_krnl_file.append('                 tapa::async_mmap<INTERFACE_WIDTH> & searchSpace,' + "\n")
    _core_krnl_file.append('                 tapa::ostream<INTERFACE_WIDTH>& load_to_compute_stream)' + "\n")
    _core_krnl_file.append('{' + "\n")
    _core_krnl_file.append('#pragma HLS INLINE OFF' + "\n")
    _core_krnl_file.append('    INTERFACE_WIDTH loaded_value = 0;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    LOAD_QUERY: ' + "\n")
    _core_krnl_file.append('    for (int i_req = 0, i_resp = 0; i_resp < (INPUT_DIM-1)/NUM_FEATURES_PER_READ + 1; ) {' + "\n")
    _core_krnl_file.append('        #pragma HLS loop_tripcount min=((INPUT_DIM-1)/NUM_FEATURES_PER_READ + 1) max=((INPUT_DIM-1)/NUM_FEATURES_PER_READ + 1)' + "\n")
    _core_krnl_file.append('        #pragma HLS pipeline II=1' + "\n")
    _core_krnl_file.append('        //Think of addr as an array index.' + "\n")
    _core_krnl_file.append('        int addr = i_req;' + "\n")
    _core_krnl_file.append('        if (i_req < (INPUT_DIM-1)/NUM_FEATURES_PER_READ + 1 && searchSpace.read_addr.try_write(addr)) {' + "\n")
    _core_krnl_file.append('            i_req++;' + "\n")
    _core_krnl_file.append('        }' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        if (!searchSpace.read_data.empty()) {' + "\n")
    _core_krnl_file.append('            loaded_value = searchSpace.read_data.read(nullptr);' + "\n")
    _core_krnl_file.append('            i_resp++;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('            // DEBUGGING:' + "\n")
    _core_krnl_file.append('            #ifndef __SYNTHESIS__' + "\n")
    _core_krnl_file.append('            for (int i = 0; i < NUM_FEATURES_PER_READ; ++i)' + "\n")
    _core_krnl_file.append('            {' + "\n")
    _core_krnl_file.append('                DATA_TYPE cur_value = 0;' + "\n")
    if (_knn_config.using_float):
        _core_krnl_file.append('                TRANSFER_TYPE tmp;' + "\n")
    elif (_knn_config.using_fixedpt):
        _core_krnl_file.append('                DATA_TYPE tmp;' + "\n")
    _core_krnl_file.append('                tmp.range(DATA_TYPE_TOTAL_SZ - 1, 0)' + "\n")
    _core_krnl_file.append('                    = loaded_value(i*DATA_TYPE_TOTAL_SZ + (DATA_TYPE_TOTAL_SZ - 1),' + "\n")
    _core_krnl_file.append('                                    i*DATA_TYPE_TOTAL_SZ);' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('                cur_value = *((DATA_TYPE*) (&tmp));' + "\n")
    _core_krnl_file.append('                if (cur_value < MAX_DATA_TYPE_VAL && debug_PE_ID == 0)' + "\n")
    _core_krnl_file.append('                {' + "\n")
    if (_knn_config.using_float):
        _core_krnl_file.append('                    printf("LOAD QUERY: value = %f, i_resp = %d\\n", cur_value, i_resp);' + "\n")
    elif (_knn_config.using_fixedpt):
        _core_krnl_file.append('                    printf("LOAD QUERY: value = %f, i_resp = %d\\n", cur_value.to_float(), i_resp);' + "\n")
    _core_krnl_file.append('                }' + "\n")
    _core_krnl_file.append('            }' + "\n")
    _core_krnl_file.append('            #endif' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('            load_to_compute_stream.write(loaded_value);' + "\n")
    _core_krnl_file.append('        }' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    LOAD_SEARCHSPACE:' + "\n")
    _core_krnl_file.append('    for (int i_req = 0, i_resp = 0; i_resp < PARTITION_LEN_IN_I; ) {' + "\n")
    _core_krnl_file.append('        #pragma HLS loop_tripcount min=PARTITION_LEN_IN_I max=PARTITION_LEN_IN_I' + "\n")
    _core_krnl_file.append('        #pragma HLS pipeline II=1' + "\n")
    _core_krnl_file.append('        //Think of addr as an array index.' + "\n")
    _core_krnl_file.append('        int addr = QUERY_DATA_RESERVE + i_req;' + "\n")
    _core_krnl_file.append('        if (i_req < PARTITION_LEN_IN_I && searchSpace.read_addr.try_write(addr)) {' + "\n")
    _core_krnl_file.append('            i_req++;' + "\n")
    _core_krnl_file.append('        }' + "\n")
    _core_krnl_file.append('        if (!searchSpace.read_data.empty()) {' + "\n")
    _core_krnl_file.append('            loaded_value = searchSpace.read_data.read(nullptr);' + "\n")
    _core_krnl_file.append('            i_resp++;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('            //// DEBUGGING: Printing the loaded data:' + "\n")
    _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
    _core_krnl_file.append('            //for (int i = 0; i < NUM_FEATURES_PER_READ; ++i)' + "\n")
    _core_krnl_file.append('            //{' + "\n")
    _core_krnl_file.append('            //    DATA_TYPE cur_value = 0;' + "\n")
    _core_krnl_file.append('            //    TRANSFER_TYPE tmp;' + "\n")
    _core_krnl_file.append('            //    tmp.range(DATA_TYPE_TOTAL_SZ - 1, 0)' + "\n")
    _core_krnl_file.append('            //        = loaded_value(i*DATA_TYPE_TOTAL_SZ + (DATA_TYPE_TOTAL_SZ - 1),' + "\n")
    _core_krnl_file.append('            //                        i*DATA_TYPE_TOTAL_SZ);' + "\n")
    _core_krnl_file.append('            //    cur_value = *((DATA_TYPE*) (&tmp));' + "\n")
    _core_krnl_file.append('            //    if (cur_value < MAX_DATA_TYPE_VAL && debug_PE_ID == 0)' + "\n")
    _core_krnl_file.append('            //    {' + "\n")
    _core_krnl_file.append('            //        printf("LOAD SEARCHSPACE: value = %f, i_resp = %d\\n", cur_value, i_resp);' + "\n")
    _core_krnl_file.append('            //    }' + "\n")
    _core_krnl_file.append('            //}' + "\n")
    _core_krnl_file.append('            //#endif' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('            //// DEBUGGING: Printing how many times we write to each stream:' + "\n")
    _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
    _core_krnl_file.append('            //if (debug_PE_ID == 0)' + "\n")
    _core_krnl_file.append('            //{' + "\n")
    _core_krnl_file.append('            //    printf("LOAD: Writing to load_to_compute_stream for the %d-th time\\n",' + "\n")
    _core_krnl_file.append('            //            i_resp-1);' + "\n")
    _core_krnl_file.append('            //}' + "\n")
    _core_krnl_file.append('            //#endif' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('            load_to_compute_stream.write(loaded_value);' + "\n")
    _core_krnl_file.append('        }' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('}' + "\n")



#############################################################
### COMPUTE
#############################################################

def GeneratePartialKNN_Compute(_core_krnl_file, _knn_config):
    _core_krnl_file.append("/*************************************************/" + "\n")
    _core_krnl_file.append("/******************* COMPUTES: *******************/" + "\n")
    _core_krnl_file.append("/*************************************************/" + "\n")
    _core_krnl_file.append("" + "\n")

    if (_knn_config.I2D_factor_w == 1):
        if (_knn_config.using_segments == 1 and _knn_config.using_Ltypes == 1):
            _core_krnl_file.append('void compute_KNN(   int debug_pe_idx,' + "\n")
            _core_krnl_file.append('                    int debug_start_idx,' + "\n")
            _core_krnl_file.append('                    tapa::istream<INTERFACE_WIDTH>&     load_to_compute_stream,' + "\n")
            _core_krnl_file.append('                    tapa::ostreams<LOCAL_DIST_DTYPE, NUM_SEGMENTS>&     compute_to_sort_stream)' + "\n")
            _core_krnl_file.append('{' + "\n")
            _core_krnl_file.append('#pragma HLS INLINE OFF' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('    int DEBUG_load_ctr = 0;' + "\n")
            _core_krnl_file.append('    #endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    INTERFACE_WIDTH cur_data = 0;' + "\n")
            _core_krnl_file.append('    DATA_TYPE local_Query[INPUT_DIM];' + "\n")
            _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_Query complete dim=1' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('    int DEBUG_write_counters[NUM_SEGMENTS] = {};' + "\n")
            _core_krnl_file.append('    #endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    /***********************************************/' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    GET_QUERYDATA:' + "\n")
            _core_krnl_file.append('    for (int i = 0 ; i < (INPUT_DIM-1)/NUM_FEATURES_PER_READ + 1; ++i)' + "\n")
            _core_krnl_file.append('    {' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('        TRANSFER_TYPE tmp = 0;' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('        DATA_TYPE tmp = 0;' + "\n")
            _core_krnl_file.append('        int input_dim_idx = 0;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        cur_data = load_to_compute_stream.read();' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        for ( int j = 0; ' + "\n")
            _core_krnl_file.append('              j < NUM_FEATURES_PER_READ && input_dim_idx < INPUT_DIM; ' + "\n")
            _core_krnl_file.append('              ++j, ++input_dim_idx) ' + "\n")
            _core_krnl_file.append('        {' + "\n")
            _core_krnl_file.append('            tmp.range(DATA_TYPE_TOTAL_SZ-1, 0)' + "\n")
            _core_krnl_file.append('                = cur_data.range(j*DATA_TYPE_TOTAL_SZ + (DATA_TYPE_TOTAL_SZ-1),' + "\n")
            _core_krnl_file.append('                                 j*DATA_TYPE_TOTAL_SZ);' + "\n")
            _core_krnl_file.append('' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('            local_Query[input_dim_idx] = *((DATA_TYPE*)(&tmp));' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('            local_Query[input_dim_idx] = tmp;' + "\n")
            _core_krnl_file.append('        }' + "\n")
            _core_krnl_file.append('    }' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    COMPUTE_DATA:' + "\n")
            _core_krnl_file.append('    for (int jj = 0; jj < SEGMENT_SIZE_IN_I; ++jj){' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        for (int ii = 0 ; ii < NUM_SEGMENTS; ++ii){' + "\n")
            _core_krnl_file.append('            #pragma HLS PIPELINE II=1' + "\n")
            _core_krnl_file.append('            LOCAL_DIST_DTYPE aggregated_local_dists = 0;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('            //if (debug_pe_idx == 0)' + "\n")
            _core_krnl_file.append('            //{' + "\n")
            _core_krnl_file.append('            //    printf("COMPUTE: Reading from load_to_compute_stream for the %d-th time\\n",' + "\n")
            _core_krnl_file.append('            //            DEBUG_load_ctr++);' + "\n")
            _core_krnl_file.append('            //}' + "\n")
            _core_krnl_file.append('            //#endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('            cur_data = load_to_compute_stream.read();' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('            for (int l2i = 0; l2i < L2I_FACTOR_W; ++l2i)' + "\n")
            _core_krnl_file.append('            {' + "\n")
            _core_krnl_file.append('            #pragma HLS UNROLL' + "\n")
            _core_krnl_file.append('                for (int d2l = 0; d2l < D2L_FACTOR_W; ++d2l){' + "\n")
            _core_krnl_file.append('                #pragma HLS UNROLL' + "\n")
            _core_krnl_file.append('                    int d2i = d2l + D2L_FACTOR_W*l2i;' + "\n")
            if (_knn_config.using_fixedpt):
                _core_krnl_file.append('                    int dist_range_idx = d2l * DATA_TYPE_TOTAL_SZ;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                    DATA_TYPE delta_squared_sum = 0.0;' + "\n")
            _core_krnl_file.append('                    int start_idx = d2i * INPUT_DIM;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                    for (int ll = 0; ll < INPUT_DIM; ++ll){' + "\n")
            _core_krnl_file.append('                        unsigned int sp_range_idx = (start_idx + ll) * DATA_TYPE_TOTAL_SZ;' + "\n")
            _core_krnl_file.append('                        DATA_TYPE sp_dim_item_value;' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('                        TRANSFER_TYPE tmp = 0;' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('                        DATA_TYPE tmp = 0;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                        tmp.range(DATA_TYPE_TOTAL_SZ-1, 0) = ' + "\n")
            _core_krnl_file.append('                            cur_data.range(sp_range_idx + (DATA_TYPE_TOTAL_SZ-1), ' + "\n")
            _core_krnl_file.append('                                                   sp_range_idx);' + "\n")
            _core_krnl_file.append('' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('                        sp_dim_item_value = *((DATA_TYPE*) (&tmp));' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('                        sp_dim_item_value = tmp;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                        #if DISTANCE_METRIC == 0 // manhattan' + "\n")
            _core_krnl_file.append('                        DATA_TYPE delta = absval(sp_dim_item_value - local_Query[ll]);' + "\n")
            _core_krnl_file.append('                        delta_squared_sum += delta;' + "\n")
            _core_krnl_file.append('                        #elif DISTANCE_METRIC == 1 // L2' + "\n")
            _core_krnl_file.append('                        DATA_TYPE delta = absval(sp_dim_item_value - local_Query[ll]);' + "\n")
            _core_krnl_file.append('                        delta_squared_sum += delta * delta;' + "\n")
            _core_krnl_file.append('                        #endif' + "\n")
            _core_krnl_file.append('                    }' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('                    aggregated_local_dists = delta_squared_sum;' + "\n")
            elif (_knn_config.using_fixedpt):
                 _core_krnl_file.append('                   aggregated_local_dists.range(dist_range_idx + (DATA_TYPE_TOTAL_SZ - 1),' + '\n')
                 _core_krnl_file.append('                                                dist_range_idx)' + '\n')
                 _core_krnl_file.append('                       = delta_squared_sum.range(DATA_TYPE_TOTAL_SZ - 1, 0);' + '\n')
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                    //#ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('                    //if (delta_squared_sum < MAX_DATA_TYPE_VAL)' + "\n")
            _core_krnl_file.append('                    //{' + "\n")
            _core_krnl_file.append('                    //    printf("COMPUTE: At index %d, delta_squared_sum = %f\\n", ' + "\n")
            _core_krnl_file.append('                    //            debug_start_idx + ii*SEGMENT_SIZE_IN_D +' + "\n")
            _core_krnl_file.append('                    //            jj*D2I_FACTOR_W + d2i,' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('                    //            delta_squared_sum);' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('                    //            delta_squared_sum.to_float());' + "\n")
            _core_krnl_file.append('                    //}' + "\n")
            _core_krnl_file.append('                    //#endif' + "\n")
            _core_krnl_file.append('                }' + "\n")
            _core_krnl_file.append('                int stream_idx = (ii*L2I_FACTOR_W + l2i)%NUM_SEGMENTS;' + "\n")
            _core_krnl_file.append('                compute_to_sort_stream[stream_idx].write(aggregated_local_dists);' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                //#ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('                //if (debug_pe_idx == 0)' + "\n")
            _core_krnl_file.append('                //{' + "\n")
            _core_krnl_file.append('                //    printf("COMPUTE: Writing the value %f to compute_to_sort_stream number %d, for the %d\'th time\\n", ' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('                //            aggregated_local_dists,' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('                //            aggregated_local_dists.to_float(),' + "\n")
            _core_krnl_file.append('                //            stream_idx,' + "\n")
            _core_krnl_file.append('                //            DEBUG_write_counters[stream_idx]++);' + "\n")
            _core_krnl_file.append('                //}' + "\n")
            _core_krnl_file.append('                //#endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                aggregated_local_dists = 0;' + "\n")
            _core_krnl_file.append('            }' + "\n")
            _core_krnl_file.append('        }' + "\n")
            _core_krnl_file.append('    }' + "\n")
            _core_krnl_file.append('}' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('' + "\n")




    elif (_knn_config.I2D_factor_w > 1):
        if ((_knn_config.using_segments == 0) and (_knn_config.using_Ltypes == 0)):
            _core_krnl_file.append('void compute_KNN(   int debug_pe_idx,' + "\n")
            _core_krnl_file.append('                    int debug_start_idx,' + "\n")
            _core_krnl_file.append('                    tapa::istream<INTERFACE_WIDTH>&     load_to_compute_stream,' + "\n")
            _core_krnl_file.append('                    tapa::ostream<DATA_TYPE>&     compute_to_sort_stream)' + "\n")
            _core_krnl_file.append('{' + "\n")
            _core_krnl_file.append('#pragma HLS INLINE OFF' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('    int DEBUG_load_ctr = 0;' + "\n")
            _core_krnl_file.append('    #endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    INTERFACE_WIDTH cur_data = 0;' + "\n")
            _core_krnl_file.append('    DATA_TYPE local_Query[INPUT_DIM];' + "\n")
            _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_Query complete dim=1' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
            _core_krnl_file.append('    int DEBUG_write_counters = {};' + "\n")
            _core_krnl_file.append('    #endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    /***********************************************/' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('    GET_QUERYDATA:' + "\n")
            _core_krnl_file.append('    for (int i = 0 ; i < CEIL_DIVISION(INPUT_DIM, NUM_FEATURES_PER_READ); ++i)' + "\n")
            _core_krnl_file.append('    {' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('        TRANSFER_TYPE tmp = 0;' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('        DATA_TYPE tmp = 0;' + "\n")
            _core_krnl_file.append('        int input_dim_idx = i*NUM_FEATURES_PER_READ;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        cur_data = load_to_compute_stream.read();' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        for ( int j = 0; ' + "\n")
            _core_krnl_file.append('              j < NUM_FEATURES_PER_READ && input_dim_idx < INPUT_DIM; ' + "\n")
            _core_krnl_file.append('              ++j, ++input_dim_idx) ' + "\n")
            _core_krnl_file.append('        {' + "\n")
            _core_krnl_file.append('            tmp.range(DATA_TYPE_TOTAL_SZ-1, 0)' + "\n")
            _core_krnl_file.append('                = cur_data.range(j*DATA_TYPE_TOTAL_SZ + (DATA_TYPE_TOTAL_SZ-1),' + "\n")
            _core_krnl_file.append('                                 j*DATA_TYPE_TOTAL_SZ);' + "\n")
            _core_krnl_file.append('' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('            local_Query[input_dim_idx] = *((DATA_TYPE*)(&tmp));' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('            local_Query[input_dim_idx] = tmp;' + "\n")
            _core_krnl_file.append('        }' + "\n")
            _core_krnl_file.append('    }' + "\n")
            _core_krnl_file.append('' + "\n")

            _core_krnl_file.append('    COMPUTE_DATA:' + "\n")
            _core_krnl_file.append('    for (int ii = 0; ii < PARTITION_LEN_IN_D; ++ii){' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('    #pragma HLS PIPELINE II=1' + "\n")
            _core_krnl_file.append('        DATA_TYPE dist_val_outer = 0;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        for (int jj = 0 ; jj < I2D_FACTOR_W; ++jj){' + "\n")
            if (_knn_config.using_fixedpt):
                _core_krnl_file.append('        #pragma HLS PIPELINE II=1' + "\n")
            _core_krnl_file.append('            cur_data = load_to_compute_stream.read();' + "\n")
            _core_krnl_file.append('            DATA_TYPE dist_val_inner = 0;' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('            for (int kk = 0; kk < NUM_FEATURES_PER_READ; ++kk) {' + "\n")
            _core_krnl_file.append('                #pragma HLS UNROLL' + "\n")
            _core_krnl_file.append('                int range_idx = kk*DATA_TYPE_TOTAL_SZ;' + "\n")
            _core_krnl_file.append('                int querypt_idx = jj*NUM_FEATURES_PER_READ + kk;' + "\n")
            _core_krnl_file.append('' + "\n")

            _core_krnl_file.append('                DATA_TYPE sp_dim_item_value;' + '\n')
            if (_knn_config.using_float):
                _core_krnl_file.append('                TRANSFER_TYPE tmp;' + '\n')
                _core_krnl_file.append('                tmp.range(DATA_TYPE_TOTAL_SZ-1, 0) =' + "\n")
                _core_krnl_file.append('                    cur_data.range(range_idx + (DATA_TYPE_TOTAL_SZ-1),' + "\n")
                _core_krnl_file.append('                                   range_idx);' + "\n")
                _core_krnl_file.append('' + '\n')
                _core_krnl_file.append('                sp_dim_item_value = *((DATA_TYPE*)(&tmp));' + '\n')
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('                sp_dim_item_value.range(DATA_TYPE_TOTAL_SZ-1, 0) =' + "\n")
                _core_krnl_file.append('                    cur_data.range(range_idx + (DATA_TYPE_TOTAL_SZ-1),' + "\n")
                _core_krnl_file.append('                                   range_idx);' + "\n")

            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('                #if DISTANCE_METRIC == 0        // Manhattan' + "\n")
            _core_krnl_file.append('                DATA_TYPE delta = absval(sp_dim_item_value - local_Query[querypt_idx]);' + "\n")
            _core_krnl_file.append('                dist_val_inner += delta;' + "\n")
            _core_krnl_file.append('                #elif DISTANCE_METRIC == 1 // L2' + "\n")
            _core_krnl_file.append('                DATA_TYPE delta = absval(sp_dim_item_value - local_Query[querypt_idx]);' + "\n")
            _core_krnl_file.append('                dist_val_inner += delta * delta;' + "\n")
            _core_krnl_file.append('                #endif' + "\n")
            _core_krnl_file.append('            }' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('            dist_val_outer += dist_val_inner;' + "\n")
            _core_krnl_file.append('        }' + "\n")
            _core_krnl_file.append('' + "\n")
            if (_knn_config.using_float):
                _core_krnl_file.append('        //#ifndef __SYNTHESIS__' + "\n")
                _core_krnl_file.append('        //if (dist_val_outer < MAX_DATA_TYPE_VAL && ii % 1000 == 0){' + "\n")
                _core_krnl_file.append('        //    printf("KDEBUG: In compute, the %d-th computed value has dist_val_outer = %f\\n", ' + "\n")
                _core_krnl_file.append('        //            ii, dist_val_outer);' + "\n")
                _core_krnl_file.append('        //}' + "\n")
                _core_krnl_file.append('        //#endif' + "\n")
            elif (_knn_config.using_fixedpt):
                _core_krnl_file.append('        //#ifndef __SYNTHESIS__' + "\n")
                _core_krnl_file.append('        //if (dist_val_outer < MAX_DATA_TYPE_VAL && ii % 1000 == 0){' + "\n")
                _core_krnl_file.append('        //    printf("KDEBUG: In compute, the %d-th computed value has dist_val_outer = %f\\n", ' + "\n")
                _core_krnl_file.append('        //            ii, dist_val_outer.to_float());' + "\n")
                _core_krnl_file.append('        //}' + "\n")
                _core_krnl_file.append('        //#endif' + "\n")
            _core_krnl_file.append('' + "\n")
            _core_krnl_file.append('        compute_to_sort_stream.write(dist_val_outer);' + "\n")
            _core_krnl_file.append('    }' + "\n")

            _core_krnl_file.append('}' + "\n")


        else: 
            _core_krnl_file.append("PYTHON SCRIPT ERROR: SOMETHING HAS GONE WRONG." + '\n') 
            print("PYTHON SCRIPT ERROR: SOMETHING HAS GONE WRONG." + '\n') 
            print("I2D = {}, USING_SEGMENTS = {}, USING_LTYPES = {}"
                .format(_knn_config.I2D_factor_w, _knn_config.using_segments, _knn_config.using_Ltypes)
            )
            sys.exit(-5)



#############################################################
### SORT STUFF
#############################################################

def __GeneratePartialKNN_SortHelpers(_core_krnl_file, _knn_config):
    ### Naive compare, naive swap
    _core_krnl_file.append('void swap(DATA_TYPE* a, DATA_TYPE* b, ' + "\n")
    _core_krnl_file.append('               int* x, int* y)' + "\n")
    _core_krnl_file.append('{' + "\n")
    _core_krnl_file.append('#pragma HLS INLINE' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    DATA_TYPE tmpdist_a;' + "\n")
    _core_krnl_file.append('    DATA_TYPE tmpdist_b;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    int tmpid_x;' + "\n")
    _core_krnl_file.append('    int tmpid_y;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    tmpdist_a = *a;' + "\n")
    _core_krnl_file.append('    tmpdist_b = *b;' + "\n")
    _core_krnl_file.append('    *b = tmpdist_a;' + "\n")
    _core_krnl_file.append('    *a = tmpdist_b;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    tmpid_x = *x;' + "\n")
    _core_krnl_file.append('    tmpid_y = *y;' + "\n")
    _core_krnl_file.append('    *x = tmpid_y;' + "\n")
    _core_krnl_file.append('    *y = tmpid_x;' + "\n")
    _core_krnl_file.append('}' + "\n")


def __GenerateSwapCalls(_core_krnl_file, _knn_config, indentation, use_D_idx):

    if (use_D_idx):
        D_idx_str = "[D_idx]"
    else:
        D_idx_str = ""

    _core_krnl_file.append(indentation + '//compare and swap odd' + '\n')
    _core_krnl_file.append(indentation + 'for(int ii=1; ii<TOP; ii+=2){' + '\n')
    _core_krnl_file.append(indentation + '#pragma HLS UNROLL' + '\n')
    _core_krnl_file.append(indentation + '#pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + '\n')
    _core_krnl_file.append(indentation + '#pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + '\n')
    _core_krnl_file.append('' + '\n')

    if (_knn_config.using_float):
        #NOTE: From testing, using DSPs on this comparison with floating point degrades sort II to 7.
        _core_krnl_file.append(indentation + '    if (local_kNearstDist' + str(D_idx_str) + '[ii] < local_kNearstDist' + str(D_idx_str) + '[ii+1]){' + '\n')
        
    elif (_knn_config.using_fixedpt):
        _core_krnl_file.append(indentation + '    ap_fixed<DATA_TYPE_TOTAL_SZ+1, DATA_TYPE_INT_PART_SZ+1, AP_RND, AP_SAT> tmp;' + '\n')
        _core_krnl_file.append(indentation + '    #pragma HLS RESOURCE variable=tmp core=AddSub_DSP' + '\n')
        _core_krnl_file.append(indentation + '    // Sign bit is 0 if positive, 1 if negative.' + '\n')
        _core_krnl_file.append(indentation + '    tmp = local_kNearstDist' + str(D_idx_str) + '[ii] - local_kNearstDist' + str(D_idx_str) + '[ii+1];' + '\n')
        _core_krnl_file.append('' + '\n')
        _core_krnl_file.append(indentation + '    if (tmp < 0){' + '\n')

    _core_krnl_file.append(indentation + '        swap(&local_kNearstDist' + str(D_idx_str) + '[ii], &local_kNearstDist' + str(D_idx_str) + '[ii+1], ' + '\n')
    _core_krnl_file.append(indentation + '                  &local_kNearstId' + str(D_idx_str) + '[ii], &local_kNearstId' + str(D_idx_str) + '[ii+1]);' + '\n')
    _core_krnl_file.append(indentation + '    }' + '\n')
    _core_krnl_file.append(indentation + '' + '\n')
    _core_krnl_file.append(indentation + '}' + '\n')

    _core_krnl_file.append('' + '\n')

    #############################################################

    _core_krnl_file.append('' + '\n')
    _core_krnl_file.append(indentation + '//compare and swap even' + '\n')
    _core_krnl_file.append(indentation + 'for(int ii=1; ii<TOP+1; ii+=2){' + '\n')
    _core_krnl_file.append(indentation + '#pragma HLS UNROLL' + '\n')
    _core_krnl_file.append(indentation + '#pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + '\n')
    _core_krnl_file.append(indentation + '#pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + '\n')
    _core_krnl_file.append('' + '\n')

    if (_knn_config.using_float):
        #NOTE: From testing, using DSPs on this comparison with floating point degrades sort II to 7.
        _core_krnl_file.append(indentation + '    if (local_kNearstDist' + str(D_idx_str) + '[ii] > local_kNearstDist' + str(D_idx_str) + '[ii-1]){' + '\n')
    elif (_knn_config.using_fixedpt):
        _core_krnl_file.append(indentation + '    ap_fixed<DATA_TYPE_TOTAL_SZ+1, DATA_TYPE_INT_PART_SZ+1, AP_RND, AP_SAT> tmp;' + '\n')
        _core_krnl_file.append(indentation + '    #pragma HLS RESOURCE variable=tmp core=AddSub_DSP' + '\n')
        _core_krnl_file.append(indentation + '    // Sign bit is 0 if positive, 1 if negative.' + '\n')
        _core_krnl_file.append(indentation + '    tmp = local_kNearstDist' + str(D_idx_str) + '[ii-1] - local_kNearstDist' + str(D_idx_str) + '[ii];' + '\n')
        _core_krnl_file.append('' + '\n')
        _core_krnl_file.append(indentation + '    if (tmp < 0){' + '\n')

    _core_krnl_file.append(indentation + '        swap(&local_kNearstDist' + str(D_idx_str) + '[ii], &local_kNearstDist' + str(D_idx_str) + '[ii-1], ' + '\n')
    _core_krnl_file.append(indentation + '                  &local_kNearstId' + str(D_idx_str) + '[ii], &local_kNearstId' + str(D_idx_str) + '[ii-1]);' + '\n')
    _core_krnl_file.append(indentation + '    }' + '\n')
    _core_krnl_file.append('' + '\n')
    _core_krnl_file.append(indentation + '}' + '\n')


def GeneratePartialKNN_Sort(_core_krnl_file, _knn_config):
    __GeneratePartialKNN_SortHelpers(_core_krnl_file, _knn_config)

    if (_knn_config.using_segments == 1 and _knn_config.using_Ltypes == 1):
        _core_krnl_file.append('void para_partial_sort(const int PE_idx,' + "\n")
        _core_krnl_file.append('                       int seg_idx,' + "\n")
        _core_krnl_file.append('                       tapa::istream<LOCAL_DIST_DTYPE>&     compute_to_sort_stream,' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('                       tapa::ostream<DATA_TYPE>&           sort_to_hiermerge_dist_stream_' + str(d2l_idx) + ',' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            if (d2l_idx == _knn_config.D2L_factor_w - 1):
                _core_krnl_file.append('                       tapa::ostream<int>&                 sort_to_hiermerge_id_stream_' + str(d2l_idx) + ')' + "\n")
            else:
                _core_krnl_file.append('                       tapa::ostream<int>&                 sort_to_hiermerge_id_stream_' + str(d2l_idx) + ',' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('#pragma HLS INLINE OFF' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    printf("SORT UNIT FOR PE #%d, SEGMENT #%d IS STARTING NOW.\\n", PE_idx, seg_idx);' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifdef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    static      // TAPA Known-issue: Static keyword fails CSIM because this is not thread-safe. ' + "\n")
        _core_krnl_file.append('                //  but when running the HW build, it will instantiate several copies of this function. So this is OK.' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    DATA_TYPE local_kNearstDist[D2L_FACTOR_W][(TOP+1)];' + "\n")
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_kNearstDist complete dim=0' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifdef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    static' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    int local_kNearstId[D2L_FACTOR_W][(TOP+1)];' + "\n")
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_kNearstId complete dim=0' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    int DEBUG_stream_counters = 0;' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    /* Our segments used to be large chunks of each partition.' + "\n")
        _core_krnl_file.append('     * Now, however, our segments are cylically split, so our ID ' + "\n")
        _core_krnl_file.append('     * logic has to change.' + "\n")
        _core_krnl_file.append('     */' + "\n")
        _core_krnl_file.append('    int start_id = PE_idx * NUM_SP_PTS_PER_KRNL_PADDED + seg_idx*D2L_FACTOR_W;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    /*******************************************/' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    // Initialize all top-K distances to MAX, and their IDs to an invalid value.' + "\n")
        _core_krnl_file.append('    INIT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int i = 0; i < D2L_FACTOR_W; ++i)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        for (int j = 0; j < TOP+1; ++j)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            local_kNearstId[i][j] = -1;' + "\n")
        _core_krnl_file.append('            local_kNearstDist[i][j] = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    SORT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int lvalue_idx = 0; lvalue_idx < (SEGMENT_SIZE_IN_L + TOP); ++lvalue_idx) {' + "\n")
        _core_krnl_file.append('    #pragma HLS PIPELINE II=2' + "\n")
        _core_krnl_file.append('        LOCAL_DIST_DTYPE cur_Lval = 0;' + "\n")
        _core_krnl_file.append('        int stream_idx = seg_idx;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        if (lvalue_idx >= SEGMENT_SIZE_IN_L) {' + "\n")

        if (_knn_config.using_float):
            _core_krnl_file.append('            cur_Lval = MAX_DATA_TYPE_VAL;' + "\n")
        elif (_knn_config.using_fixedpt):
            _core_krnl_file.append('            for (int D_idx = 0; D_idx < D2L_FACTOR_W; ++D_idx)' + "\n")
            _core_krnl_file.append('            {' + "\n")
            _core_krnl_file.append('                DATA_TYPE cur_Dval = MAX_DATA_TYPE_VAL;' + "\n")
            _core_krnl_file.append('                cur_Lval.range(DATA_TYPE_TOTAL_SZ*(D_idx+1) - 1, DATA_TYPE_TOTAL_SZ*(D_idx)) =' + "\n")
            _core_krnl_file.append('                    cur_Dval.range(DATA_TYPE_TOTAL_SZ-1, 0);' + "\n")
            _core_krnl_file.append('            }' + "\n")

        _core_krnl_file.append('        } else {' + "\n")
        _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('            //if (PE_idx == 0)' + "\n")
        _core_krnl_file.append('            //{' + "\n")
        _core_krnl_file.append('            //    printf("PPS Unit %d: Reading from compute_to_sort_stream %d, for the %d-th time\\n", ' + "\n")
        _core_krnl_file.append('            //            seg_idx,' + "\n")
        _core_krnl_file.append('            //            stream_idx,' + "\n")
        _core_krnl_file.append('            //            DEBUG_stream_counters++);' + "\n")
        _core_krnl_file.append('            //}' + "\n")
        _core_krnl_file.append('            //#endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            cur_Lval = compute_to_sort_stream.read();' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        for (int D_idx = 0; D_idx < D2L_FACTOR_W; ++D_idx)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('        #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            unsigned char range_idx = (D_idx)*DATA_TYPE_TOTAL_SZ;' + "\n")
        _core_krnl_file.append('            DATA_TYPE cur_Dval;' + "\n")
        _core_krnl_file.append('' + "\n")
        if (_knn_config.using_float):
            _core_krnl_file.append('            cur_Dval = cur_Lval;' + "\n")
        elif (_knn_config.using_fixedpt):
            _core_krnl_file.append('            cur_Dval.range(DATA_TYPE_TOTAL_SZ - 1, 0) = ' + "\n")
            _core_krnl_file.append('                cur_Lval.range(range_idx + (DATA_TYPE_TOTAL_SZ - 1), range_idx);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            local_kNearstDist[D_idx][0] = cur_Dval;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            local_kNearstId[D_idx][0] = start_id + lvalue_idx*D2L_FACTOR_W*NUM_SEGMENTS + D_idx;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('            //printf("SORT: Current ID = %d, cur_Dval = %f\\n", start_id + lvalue_idx, cur_Dval);' + "\n")
        _core_krnl_file.append('            //printf("SORT: Best ID = %d, Best Dval = %f\\n", local_kNearstId[D_idx][TOP], local_kNearstDist[D_idx][TOP]);' + "\n")
        _core_krnl_file.append('            //#endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            //compare and swap odd' + "\n")
        _core_krnl_file.append('            for(int ii=1; ii<TOP; ii+=2){' + "\n")
        _core_krnl_file.append('            #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('                if (local_kNearstDist[D_idx][ii] < local_kNearstDist[D_idx][ii+1]){' + "\n")
        _core_krnl_file.append('                    swap(&local_kNearstDist[D_idx][ii], &local_kNearstDist[D_idx][ii+1], ' + "\n")
        _core_krnl_file.append('                              &local_kNearstId[D_idx][ii], &local_kNearstId[D_idx][ii+1]);' + "\n")
        _core_krnl_file.append('                }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            //compare and swap even' + "\n")
        _core_krnl_file.append('            for(int ii=1; ii<TOP+1; ii+=2){' + "\n")
        _core_krnl_file.append('            #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('                if (local_kNearstDist[D_idx][ii] > local_kNearstDist[D_idx][ii-1]){' + "\n")
        _core_krnl_file.append('                    swap(&local_kNearstDist[D_idx][ii], &local_kNearstDist[D_idx][ii-1], ' + "\n")
        _core_krnl_file.append('                              &local_kNearstId[D_idx][ii], &local_kNearstId[D_idx][ii-1]);' + "\n")
        _core_krnl_file.append('                }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    // Write data out' + "\n")

        _core_krnl_file.append('    OUTPUT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int j = TOP; j > 0; --j)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS PIPELINE II=1' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('        sort_to_hiermerge_dist_stream_' + str(d2l_idx) + '.write(local_kNearstDist[' + str(d2l_idx) + '][j]);' + "\n")
            _core_krnl_file.append('        sort_to_hiermerge_id_stream_' + str(d2l_idx) + '.write(local_kNearstId[' + str(d2l_idx) + '][j]);' + "\n")
        _core_krnl_file.append('    }' + "\n")




        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    for (int i = 0; i < D2L_FACTOR_W; ++i)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        for (int j = 0; j < TOP+1; ++j)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            printf("AFTER SORT: local_kNearst[%3d][%3d][%3d]:\\n", seg_idx, i, j);' + "\n")
        if (_knn_config.using_float):
            _core_krnl_file.append('            printf("AFTER SORT:     Dist = %5.10f\\n",   local_kNearstDist[i][j]);' + "\n")
        if (_knn_config.using_fixedpt):
            _core_krnl_file.append('            printf("AFTER SORT:     Dist = %5.10f\\n",   local_kNearstDist[i][j].to_float());' + "\n")
        _core_krnl_file.append('            printf("AFTER SORT:     Id = %d\\n",         local_kNearstId[i][j]);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        printf("\\n");' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    REINITIALIZATION_LOOP:' + "\n")
        _core_krnl_file.append('    for (int i = 0; i < D2L_FACTOR_W; ++i){' + "\n")
        _core_krnl_file.append('        for (int j = 0; j < TOP+1; ++j){' + "\n")
        _core_krnl_file.append('        #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('            // Reset the kNearst values so we can run the next iteration.' + "\n")
        _core_krnl_file.append('            local_kNearstId[i][j] = -1;' + "\n")
        _core_krnl_file.append('            local_kNearstDist[i][j] = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")



    elif ((_knn_config.using_segments == 0) and (_knn_config.using_Ltypes == 1)):
        _core_krnl_file.append('void para_partial_sort(const int PE_idx,' + "\n")
        _core_krnl_file.append('                       tapa::istream<LOCAL_DIST_DTYPE>&     compute_to_sort_stream,' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('                       tapa::ostream<DATA_TYPE>&           sort_to_hiermerge_dist_stream_' + str(d2l_idx) + ',' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            if (d2l_idx == _knn_config.D2L_factor_w - 1):
                _core_krnl_file.append('                       tapa::ostream<int>&                 sort_to_hiermerge_id_stream_' + str(d2l_idx) + ')' + "\n")
            else:
                _core_krnl_file.append('                       tapa::ostream<int>&                 sort_to_hiermerge_id_stream_' + str(d2l_idx) + ',' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('#pragma HLS INLINE OFF' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    printf("SORT UNIT FOR PE #%d IS STARTING NOW.\\n", PE_idx);' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifdef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    static      // TAPA Known-issue: Static keyword fails CSIM because this is not thread-safe. ' + "\n")
        _core_krnl_file.append('                //  but when running the HW build, it will instantiate several copies of this function. So this is OK.' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    DATA_TYPE local_kNearstDist[D2L_FACTOR_W][(TOP+1)];' + "\n")
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_kNearstDist complete dim=0' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifdef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    static' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    int local_kNearstId[D2L_FACTOR_W][(TOP+1)];' + "\n")
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_kNearstId complete dim=0' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    int DEBUG_stream_counters = 0;' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    int start_id = PE_idx * NUM_SP_PTS_PER_KRNL_PADDED;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    /*******************************************/' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    // Initialize all top-K distances to MAX, and their IDs to an invalid value.' + "\n")
        _core_krnl_file.append('    INIT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int i = 0; i < D2L_FACTOR_W; ++i)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        for (int j = 0; j < TOP+1; ++j)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            local_kNearstId[i][j] = -1;' + "\n")
        _core_krnl_file.append('            local_kNearstDist[i][j] = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    SORT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int lvalue_idx = 0; lvalue_idx < (PARTITION_LEN_IN_L + TOP); ++lvalue_idx) {' + "\n")
        _core_krnl_file.append('    #pragma HLS PIPELINE II=2' + "\n")
        _core_krnl_file.append('        LOCAL_DIST_DTYPE cur_Lval = 0;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        if (lvalue_idx >= PARTITION_LEN_IN_L) {' + "\n")

        if (_knn_config.using_float):
            _core_krnl_file.append('            cur_Lval = MAX_DATA_TYPE_VAL;' + "\n")
        elif (_knn_config.using_fixedpt):
            _core_krnl_file.append('            for (int D_idx = 0; D_idx < D2L_FACTOR_W; ++D_idx)' + "\n")
            _core_krnl_file.append('            {' + "\n")
            _core_krnl_file.append('                DATA_TYPE cur_Dval = MAX_DATA_TYPE_VAL;' + "\n")
            _core_krnl_file.append('                cur_Lval.range(DATA_TYPE_TOTAL_SZ*(D_idx+1) - 1, DATA_TYPE_TOTAL_SZ*(D_idx)) =' + "\n")
            _core_krnl_file.append('                    cur_Dval.range(DATA_TYPE_TOTAL_SZ-1, 0);' + "\n")
            _core_krnl_file.append('            }' + "\n")

        _core_krnl_file.append('        } else {' + "\n")
        _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('            //printf("PPS Unit for PE %d: Reading from compute_to_sort_stream, for the %d-th time\\n", ' + "\n")
        _core_krnl_file.append('            //        PE_idx,' + "\n")
        _core_krnl_file.append('            //        DEBUG_stream_counters++);' + "\n")
        _core_krnl_file.append('            //#endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            cur_Lval = compute_to_sort_stream.read();' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        for (int D_idx = 0; D_idx < D2L_FACTOR_W; ++D_idx)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('        #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            unsigned char range_idx = (D_idx)*DATA_TYPE_TOTAL_SZ;' + "\n")
        _core_krnl_file.append('            DATA_TYPE cur_Dval;' + "\n")
        _core_krnl_file.append('' + "\n")
        if (_knn_config.using_float):
            _core_krnl_file.append('            cur_Dval = cur_Lval;' + "\n")
        elif (_knn_config.using_fixedpt):
            _core_krnl_file.append('            cur_Dval.range(DATA_TYPE_TOTAL_SZ - 1, 0) = ' + "\n")
            _core_krnl_file.append('                cur_Lval.range(range_idx + (DATA_TYPE_TOTAL_SZ - 1), range_idx);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            local_kNearstDist[D_idx][0] = cur_Dval;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            local_kNearstId[D_idx][0] = start_id + lvalue_idx*D2L_FACTOR_W + D_idx;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('            //printf("SORT: Current ID = %d, cur_Dval = %f\\n", start_id + lvalue_idx, cur_Dval);' + "\n")
        _core_krnl_file.append('            //printf("SORT: Best ID = %d, Best Dval = %f\\n", local_kNearstId[D_idx][TOP], local_kNearstDist[D_idx][TOP]);' + "\n")
        _core_krnl_file.append('            //#endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            //compare and swap odd' + "\n")
        _core_krnl_file.append('            for(int ii=1; ii<TOP; ii+=2){' + "\n")
        _core_krnl_file.append('            #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('                if (local_kNearstDist[D_idx][ii] < local_kNearstDist[D_idx][ii+1]){' + "\n")
        _core_krnl_file.append('                    swap(&local_kNearstDist[D_idx][ii], &local_kNearstDist[D_idx][ii+1], ' + "\n")
        _core_krnl_file.append('                              &local_kNearstId[D_idx][ii], &local_kNearstId[D_idx][ii+1]);' + "\n")
        _core_krnl_file.append('                }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            //compare and swap even' + "\n")
        _core_krnl_file.append('            for(int ii=1; ii<TOP+1; ii+=2){' + "\n")
        _core_krnl_file.append('            #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + "\n")
        _core_krnl_file.append('            #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('                if (local_kNearstDist[D_idx][ii] > local_kNearstDist[D_idx][ii-1]){' + "\n")
        _core_krnl_file.append('                    swap(&local_kNearstDist[D_idx][ii], &local_kNearstDist[D_idx][ii-1], ' + "\n")
        _core_krnl_file.append('                              &local_kNearstId[D_idx][ii], &local_kNearstId[D_idx][ii-1]);' + "\n")
        _core_krnl_file.append('                }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    // Write data out' + "\n")

        _core_krnl_file.append('    OUTPUT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int j = TOP; j > 0; --j)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS PIPELINE II=1' + "\n")
        for d2l_idx in range(_knn_config.D2L_factor_w):
            _core_krnl_file.append('        sort_to_hiermerge_dist_stream_' + str(d2l_idx) + '.write(local_kNearstDist[' + str(d2l_idx) + '][j]);' + "\n")
            _core_krnl_file.append('        sort_to_hiermerge_id_stream_' + str(d2l_idx) + '.write(local_kNearstId[' + str(d2l_idx) + '][j]);' + "\n")
        _core_krnl_file.append('    }' + "\n")

        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    for (int i = 0; i < D2L_FACTOR_W; ++i)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        for (int j = 0; j < TOP+1; ++j)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            printf("AFTER SORT: local_kNearst[%3d][%3d]:\\n", i, j);' + "\n")
        if (_knn_config.using_float):
            _core_krnl_file.append('            printf("AFTER SORT:     Dist = %5.10f\\n",   local_kNearstDist[i][j]);' + "\n")
        if (_knn_config.using_fixedpt):
            _core_krnl_file.append('            printf("AFTER SORT:     Dist = %5.10f\\n",   local_kNearstDist[i][j].to_float());' + "\n")
        _core_krnl_file.append('            printf("AFTER SORT:     Id = %d\\n",         local_kNearstId[i][j]);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        printf("\\n");' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    REINITIALIZATION_LOOP:' + "\n")
        _core_krnl_file.append('    for (int i = 0; i < D2L_FACTOR_W; ++i){' + "\n")
        _core_krnl_file.append('        for (int j = 0; j < TOP+1; ++j){' + "\n")
        _core_krnl_file.append('        #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('            // Reset the kNearst values so we can run the next iteration.' + "\n")
        _core_krnl_file.append('            local_kNearstId[i][j] = -1;' + "\n")
        _core_krnl_file.append('            local_kNearstDist[i][j] = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")



    elif ((_knn_config.using_segments == 0) and (_knn_config.using_Ltypes == 0)):
        _core_krnl_file.append('void para_partial_sort(const int PE_idx,' + "\n")
        _core_krnl_file.append('                       tapa::istream<DATA_TYPE>&     compute_to_sort_stream,' + "\n")
        _core_krnl_file.append('                       tapa::ostream<DATA_TYPE>&     sort_to_hiermerge_dist_stream,' + "\n")
        _core_krnl_file.append('                       tapa::ostream<int>&           sort_to_hiermerge_id_stream)' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('#pragma HLS INLINE OFF' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    printf("SORT UNIT FOR PE #%d IS STARTING NOW.\\n", PE_idx);' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifdef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    static      // TAPA Known-issue: Static keyword fails CSIM because this is not thread-safe. ' + "\n")
        _core_krnl_file.append('                //  but when running the HW build, it will instantiate several copies of this function. So this is OK.' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    DATA_TYPE local_kNearstDist[(TOP+1)];' + "\n")
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_kNearstDist complete dim=0' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifdef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    static' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    int local_kNearstId[(TOP+1)];' + "\n")
        _core_krnl_file.append('    #pragma HLS ARRAY_PARTITION variable=local_kNearstId complete dim=0' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    int DEBUG_stream_counters = 0;' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    int start_id = PE_idx * NUM_SP_PTS_PER_KRNL_PADDED;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    /*******************************************/' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    // Initialize all top-K distances to MAX, and their IDs to an invalid value.' + "\n")
        _core_krnl_file.append('    INIT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int j = 0; j < TOP+1; ++j)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        local_kNearstId[j] = -1;' + "\n")
        _core_krnl_file.append('        local_kNearstDist[j] = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")

        _core_krnl_file.append('    SORT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int dvalue_idx = 0; dvalue_idx < (PARTITION_LEN_IN_D + TOP); ++dvalue_idx) {' + "\n")
        _core_krnl_file.append('    #pragma HLS PIPELINE II=I2D_FACTOR_W' + "\n")
        _core_krnl_file.append('        DATA_TYPE cur_Dval = 0;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        if (dvalue_idx >= PARTITION_LEN_IN_D) {' + "\n")
        _core_krnl_file.append('            cur_Dval = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('        } else {' + "\n")
        _core_krnl_file.append('            //#ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('            //printf("PPS Unit for PE %d: Reading from compute_to_sort_stream, for the %d-th time\\n", ' + "\n")
        _core_krnl_file.append('            //        PE_idx,' + "\n")
        _core_krnl_file.append('            //        DEBUG_stream_counters++);' + "\n")
        _core_krnl_file.append('            //#endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            cur_Dval = compute_to_sort_stream.read();' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        local_kNearstDist[0] = cur_Dval;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        local_kNearstId[0] = start_id + dvalue_idx;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        //#ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('        //printf("SORT: Current ID = %d, cur_Dval = %f\\n", start_id + dvalue_idx, cur_Dval);' + "\n")
        _core_krnl_file.append('        //printf("SORT: Best ID = %d, Best Dval = %f\\n", local_kNearstId[TOP], local_kNearstDist[TOP]);' + "\n")
        _core_krnl_file.append('        //#endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        //compare and swap odd' + "\n")
        _core_krnl_file.append('        for(int ii=1; ii<TOP; ii+=2){' + "\n")
        _core_krnl_file.append('        #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('        #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + "\n")
        _core_krnl_file.append('        #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (local_kNearstDist[ii] < local_kNearstDist[ii+1]){' + "\n")
        _core_krnl_file.append('                swap(&local_kNearstDist[ii], &local_kNearstDist[ii+1], ' + "\n")
        _core_krnl_file.append('                          &local_kNearstId[ii], &local_kNearstId[ii+1]);' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        //compare and swap even' + "\n")
        _core_krnl_file.append('        for(int ii=1; ii<TOP+1; ii+=2){' + "\n")
        _core_krnl_file.append('        #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('        #pragma HLS DEPENDENCE variable="local_kNearstDist" inter false' + "\n")
        _core_krnl_file.append('        #pragma HLS DEPENDENCE variable="local_kNearstId" inter false' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (local_kNearstDist[ii] > local_kNearstDist[ii-1]){' + "\n")
        _core_krnl_file.append('                swap(&local_kNearstDist[ii], &local_kNearstDist[ii-1], ' + "\n")
        _core_krnl_file.append('                          &local_kNearstId[ii], &local_kNearstId[ii-1]);' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")

        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    // Write data out' + "\n")
        _core_krnl_file.append('    OUTPUT_LOOP:' + "\n")
        _core_krnl_file.append('    for (int j = TOP; j > 0; --j)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS PIPELINE II=1' + "\n")
        _core_krnl_file.append('        sort_to_hiermerge_dist_stream.write(local_kNearstDist[j]);' + "\n")
        _core_krnl_file.append('        sort_to_hiermerge_id_stream.write(local_kNearstId[j]);' + "\n")
        _core_krnl_file.append('    }' + "\n")

        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    for (int j = 0; j < TOP+1; ++j)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        printf("AFTER SORT: local_kNearst[%3d]:\\n", j);' + "\n")
        if (_knn_config.using_float):
            _core_krnl_file.append('        printf("AFTER SORT:     Dist = %5.10f\\n",   local_kNearstDist[j]);' + "\n")
        if (_knn_config.using_fixedpt):
            _core_krnl_file.append('        printf("AFTER SORT:     Dist = %5.10f\\n",   local_kNearstDist[j].to_float());' + "\n")
        _core_krnl_file.append('        printf("AFTER SORT:     Id = %d\\n",         local_kNearstId[j]);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    REINITIALIZATION_LOOP:' + "\n")
        _core_krnl_file.append('    for (int j = 0; j < TOP+1; ++j){' + "\n")
        _core_krnl_file.append('    #pragma HLS UNROLL' + "\n")
        _core_krnl_file.append('        // Reset the kNearst values so we can run the next iteration.' + "\n")
        _core_krnl_file.append('        local_kNearstId[j] = -1;' + "\n")
        _core_krnl_file.append('        local_kNearstDist[j] = MAX_DATA_TYPE_VAL;' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")

    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")


def _generate_write_out_mmap(_core_krnl_file, _knn_config):
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('void write_out_mmap(' + "\n")
    _core_krnl_file.append('                    tapa::async_mmap<INT32>&    output_knn,' + "\n")
    _core_krnl_file.append('                    DATA_TYPE                   output_dist,' + "\n")
    _core_krnl_file.append('                    int                         output_id,' + "\n")
    _core_krnl_file.append('                    int&                        i_req_output, ' + "\n")
    _core_krnl_file.append('                    int&                        i_resp_output' + "\n")
    _core_krnl_file.append(') {' + "\n")
    _core_krnl_file.append('#pragma HLS INLINE' + "\n")
    _core_krnl_file.append('    INT32 outval = 0;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    if (i_req_output < 2*TOP && i_req_output >= 0 && ' + "\n")
    _core_krnl_file.append('        !output_knn.write_addr.full() && ' + "\n")
    _core_krnl_file.append('        !output_knn.write_data.full()' + "\n")
    _core_krnl_file.append('    ) {' + "\n")

    if (_knn_config.using_float):
        _core_krnl_file.append('        outval = *(INT32*) &output_dist;' + "\n")
    else:
        _core_krnl_file.append('        outval.range(DATA_TYPE_TOTAL_SZ-1, 0) = output_dist.range(DATA_TYPE_TOTAL_SZ-1, 0);' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        //#ifndef __SYNTHESIS__' + "\n")
    _core_krnl_file.append('        //printf("KDEBUG: i_req_output = %d. Outval = %f, output_dist = %f.\\n",' + "\n")
    _core_krnl_file.append('        //        i_req_output, outval.to_float(), output_dist.to_float());' + "\n")
    _core_krnl_file.append('        //#endif' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        output_knn.write_addr.try_write(i_req_output);' + "\n")
    _core_krnl_file.append('        output_knn.write_data.try_write(outval);' + "\n")
    _core_krnl_file.append('        --i_req_output;' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    if (!output_knn.write_resp.empty()) {' + "\n")
    _core_krnl_file.append('        i_resp_output += (unsigned int) (output_knn.write_resp.read(nullptr)) + 1;' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    if (i_req_output < 2*TOP && i_req_output >= 0 && ' + "\n")
    _core_krnl_file.append('        !output_knn.write_addr.full() && ' + "\n")
    _core_krnl_file.append('        !output_knn.write_data.full()' + "\n")
    _core_krnl_file.append('    ) {' + "\n")
    _core_krnl_file.append('        outval = * (INT32*) &output_id;' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        //#ifndef __SYNTHESIS__' + "\n")
    _core_krnl_file.append('        //printf("KDEBUG: i_req_output = %d. Outval = %f, output_id = %d\\n",' + "\n")
    _core_krnl_file.append('        //        i_req_output, outval.to_float(), output_id);' + "\n")
    _core_krnl_file.append('        //#endif' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        output_knn.write_addr.try_write(i_req_output);' + "\n")
    _core_krnl_file.append('        output_knn.write_data.try_write(outval);' + "\n")
    _core_krnl_file.append('        --i_req_output;' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    if (!output_knn.write_resp.empty()) {' + "\n")
    _core_krnl_file.append('        i_resp_output += (unsigned int) (output_knn.write_resp.read(nullptr)) + 1;' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('}' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")





def GeneratePartialKNN_HierMerge(_core_krnl_file, _knn_config):
    _generate_write_out_mmap(_core_krnl_file, _knn_config)

    if ((_knn_config.num_PE > 1) or (_knn_config.using_segments == 1) or (_knn_config.using_Ltypes == 1)):
        _core_krnl_file.append('void merge_dual_streams(' + "\n")
        _core_krnl_file.append('                        int debug_PE_idx,' + "\n")
        _core_krnl_file.append('                        int debug_seg_d2l_idx,' + "\n")
        _core_krnl_file.append('                        int debug_stage_idx,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::ostream<DATA_TYPE>&   hiermerge_dist_ostream,' + "\n")
        _core_krnl_file.append('                        tapa::ostream<int>&         hiermerge_id_ostream' + "\n")
        _core_krnl_file.append(')' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    printf("NOTE: USING MONOMERGE!\\n");' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('    int id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    int id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    int stream1_read_count = 1;' + "\n")
        _core_krnl_file.append('    int stream2_read_count = 1;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    for (int k = TOP-1; k > 0; --k)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('        if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            printf("KDEBUG: Hiermerge for PE %d, STAGE %d, seg_d2l = %d, stream1_read_count = %d, stream2_read_count = %d\\n", ' + "\n")
        _core_krnl_file.append('                    debug_PE_idx, debug_stage_idx, debug_seg_d2l_idx,' + "\n")
        _core_krnl_file.append('                    stream1_read_count, stream2_read_count);' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        #endif' + "\n")
        _core_krnl_file.append('        if (dist_1 <= dist_2)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            hiermerge_dist_ostream.write(dist_1);' + "\n")
        _core_krnl_file.append('            hiermerge_id_ostream.write(id_1);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream1_read_count;' + "\n")
        _core_krnl_file.append('                dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('                id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        else' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            hiermerge_dist_ostream.write(dist_2);' + "\n")
        _core_krnl_file.append('            hiermerge_id_ostream.write(id_2);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream2_read_count;' + "\n")
        _core_krnl_file.append('                dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('                id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    // Final write.' + "\n")
        _core_krnl_file.append('    if (dist_1 <= dist_2) {' + "\n")
        _core_krnl_file.append('        hiermerge_dist_ostream.write(dist_1);' + "\n")
        _core_krnl_file.append('        hiermerge_id_ostream.write(id_1);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    else {' + "\n")
        _core_krnl_file.append('        hiermerge_dist_ostream.write(dist_2);' + "\n")
        _core_krnl_file.append('        hiermerge_id_ostream.write(id_2);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        printf("KDEBUG: Hiermerge for PE %d, STAGE #%d, seg_d2l = %d, Emptying the input FIFOs now...\\n",' + "\n")
        _core_krnl_file.append('                debug_PE_idx, debug_stage_idx, debug_seg_d2l_idx);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    // Empty the input streams.' + "\n")
        _core_krnl_file.append('    // NOTE: The total tripcount of these loops will be TOP.' + "\n")
        _core_krnl_file.append('    while (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream1_read_count;' + "\n")
        _core_krnl_file.append('        dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('        id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    while (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream2_read_count;' + "\n")
        _core_krnl_file.append('        dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('        id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('void merge_trio_streams(' + "\n")
        _core_krnl_file.append('                        int debug_PE_idx,' + "\n")
        _core_krnl_file.append('                        int debug_seg_d2l_idx,' + "\n")
        _core_krnl_file.append('                        int debug_stage_idx,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_3,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_3,' + "\n")
        _core_krnl_file.append('                        tapa::ostream<DATA_TYPE>&   hiermerge_dist_ostream,' + "\n")
        _core_krnl_file.append('                        tapa::ostream<int>&         hiermerge_id_ostream' + "\n")
        _core_krnl_file.append(')' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_3 = hiermerge_dist_istream_3.read();' + "\n")
        _core_krnl_file.append('    int id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    int id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    int id_3 = hiermerge_id_istream_3.read();' + "\n")
        _core_krnl_file.append('    int stream1_read_count = 1;' + "\n")
        _core_krnl_file.append('    int stream2_read_count = 1;' + "\n")
        _core_krnl_file.append('    int stream3_read_count = 1;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    for (int k = TOP-1; k > 0; --k)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('        if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            printf("KDEBUG: Hiermerge for PE %d, STAGE %d, seg_d2l = %d, stream1_read_count = %d, stream2_read_count = %d, stream3_read_count = %d\\n", ' + "\n")
        _core_krnl_file.append('                    debug_PE_idx, debug_stage_idx, debug_seg_d2l_idx,' + "\n")
        _core_krnl_file.append('                    stream1_read_count, stream2_read_count, stream3_read_count);' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        if ( (dist_1 <= dist_2) && (dist_1 <= dist_3) )' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            hiermerge_dist_ostream.write(dist_1);' + "\n")
        _core_krnl_file.append('            hiermerge_id_ostream.write(id_1);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream1_read_count;' + "\n")
        _core_krnl_file.append('                dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('                id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        else if ( (dist_2 <= dist_3) && (dist_2 <= dist_1) )' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            hiermerge_dist_ostream.write(dist_2);' + "\n")
        _core_krnl_file.append('            hiermerge_id_ostream.write(id_2);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream2_read_count;' + "\n")
        _core_krnl_file.append('                dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('                id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        else' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            hiermerge_dist_ostream.write(dist_3);' + "\n")
        _core_krnl_file.append('            hiermerge_id_ostream.write(id_3);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream3_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream3_read_count;' + "\n")
        _core_krnl_file.append('                dist_3 = hiermerge_dist_istream_3.read();' + "\n")
        _core_krnl_file.append('                id_3 = hiermerge_id_istream_3.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    // Final write.' + "\n")
        _core_krnl_file.append('    if ( (dist_1 <= dist_2) && (dist_1 <= dist_3) ){' + "\n")
        _core_krnl_file.append('        hiermerge_dist_ostream.write(dist_1);' + "\n")
        _core_krnl_file.append('        hiermerge_id_ostream.write(id_1);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    else if ( (dist_2 <= dist_3) && (dist_2 <= dist_1) ){' + "\n")
        _core_krnl_file.append('        hiermerge_dist_ostream.write(dist_2);' + "\n")
        _core_krnl_file.append('        hiermerge_id_ostream.write(id_2);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    else{' + "\n")
        _core_krnl_file.append('        hiermerge_dist_ostream.write(dist_3);' + "\n")
        _core_krnl_file.append('        hiermerge_id_ostream.write(id_3);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        printf("KDEBUG: Hiermerge for PE %d, STAGE #%d, seg_d2l = %d, Emptying the input FIFOs now...\\n",' + "\n")
        _core_krnl_file.append('                debug_PE_idx, debug_stage_idx, debug_seg_d2l_idx);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    // Empty the input streams.' + "\n")
        _core_krnl_file.append('    // NOTE: The total tripcount of these loops will be 2*TOP.' + "\n")
        _core_krnl_file.append('    while (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream1_read_count;' + "\n")
        _core_krnl_file.append('        dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('        id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    while (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream2_read_count;' + "\n")
        _core_krnl_file.append('        dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('        id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    while (stream3_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP max=TOP' + "\n")
        _core_krnl_file.append('        ++stream3_read_count;' + "\n")
        _core_krnl_file.append('        dist_3 = hiermerge_dist_istream_3.read();' + "\n")
        _core_krnl_file.append('        id_3 = hiermerge_id_istream_3.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")
        _core_krnl_file.append('' + "\n")

        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('void merge_dual_streams_FINAL(' + "\n")
        _core_krnl_file.append('                        int debug_PE_idx,' + "\n")
        _core_krnl_file.append('                        int debug_seg_d2l_idx,' + "\n")
        _core_krnl_file.append('                        int debug_stage_idx,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::async_mmap<INT32>&    hiermerge_output' + "\n")
        _core_krnl_file.append(')' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('    int id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    int id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    int stream1_read_count = 1;' + "\n")
        _core_krnl_file.append('    int stream2_read_count = 1;' + "\n")
        _core_krnl_file.append('    int i_req_output  = 2*TOP-1;' + "\n")
        _core_krnl_file.append('    int i_resp_output = 2*TOP-1;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    for (int k = TOP-1; k > 0; --k)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('        if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            printf("KDEBUG: FINAL Hiermerge, stream1_read_count = %d, stream2_read_count = %d\\n", ' + "\n")
        _core_krnl_file.append('                    stream1_read_count, stream2_read_count);' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        #endif' + "\n")
        _core_krnl_file.append('        if (dist_1 <= dist_2)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_1,' + "\n")
        _core_krnl_file.append('                            id_1,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream1_read_count;' + "\n")
        _core_krnl_file.append('                dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('                id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        else' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_2,' + "\n")
        _core_krnl_file.append('                            id_2,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream2_read_count;' + "\n")
        _core_krnl_file.append('                dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('                id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    // Final write.' + "\n")
        _core_krnl_file.append('    if (dist_1 <= dist_2) {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_1,' + "\n")
        _core_krnl_file.append('                            id_1,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    else {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_2,' + "\n")
        _core_krnl_file.append('                            id_2,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        printf("KDEBUG: FINAL Hiermerge, Emptying the input FIFOs now...\\n");' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    // Empty the input streams.' + "\n")
        _core_krnl_file.append('    while (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream1_read_count;' + "\n")
        _core_krnl_file.append('        dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('        id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    while (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream2_read_count;' + "\n")
        _core_krnl_file.append('        dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('        id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('void merge_trio_streams_FINAL(' + "\n")
        _core_krnl_file.append('                        int debug_PE_idx,' + "\n")
        _core_krnl_file.append('                        int debug_seg_d2l_idx,' + "\n")
        _core_krnl_file.append('                        int debug_stage_idx,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_1,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_2,' + "\n")
        _core_krnl_file.append('                        tapa::istream<DATA_TYPE>&   hiermerge_dist_istream_3,' + "\n")
        _core_krnl_file.append('                        tapa::istream<int>&         hiermerge_id_istream_3,' + "\n")
        _core_krnl_file.append('                        tapa::async_mmap<INT32>&    hiermerge_output' + "\n")
        _core_krnl_file.append(')' + "\n")
        _core_krnl_file.append('{' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('    DATA_TYPE dist_3 = hiermerge_dist_istream_3.read();' + "\n")
        _core_krnl_file.append('    int id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    int id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    int id_3 = hiermerge_id_istream_3.read();' + "\n")
        _core_krnl_file.append('    int stream1_read_count = 1;' + "\n")
        _core_krnl_file.append('    int stream2_read_count = 1;' + "\n")
        _core_krnl_file.append('    int stream3_read_count = 1;' + "\n")
        _core_krnl_file.append('    int i_req_output  = 2*TOP-1;' + "\n")
        _core_krnl_file.append('    int i_resp_output = 2*TOP-1;' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    for (int k = TOP-1; k > 0; --k)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('        if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            printf("KDEBUG: FINAL Hiermerge, stream1_read_count = %d, stream2_read_count = %d, stream3_read_count = %d\\n", ' + "\n")
        _core_krnl_file.append('                    stream1_read_count, stream2_read_count, stream3_read_count);' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        #endif' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        if ( (dist_1 <= dist_2) && (dist_1 <= dist_3) )' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_1,' + "\n")
        _core_krnl_file.append('                            id_1,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream1_read_count;' + "\n")
        _core_krnl_file.append('                dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('                id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('        else if ( (dist_2 <= dist_3) && (dist_2 <= dist_1) )' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_2,' + "\n")
        _core_krnl_file.append('                            id_2,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream2_read_count;' + "\n")
        _core_krnl_file.append('                dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('                id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('        else' + "\n")
        _core_krnl_file.append('        {' + "\n")
        _core_krnl_file.append('            write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                            dist_3,' + "\n")
        _core_krnl_file.append('                            id_3,' + "\n")
        _core_krnl_file.append('                            i_req_output, ' + "\n")
        _core_krnl_file.append('                            i_resp_output);' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('            if (stream3_read_count < TOP)' + "\n")
        _core_krnl_file.append('            {' + "\n")
        _core_krnl_file.append('                ++stream3_read_count;' + "\n")
        _core_krnl_file.append('                dist_3 = hiermerge_dist_istream_3.read();' + "\n")
        _core_krnl_file.append('                id_3 = hiermerge_id_istream_3.read();' + "\n")
        _core_krnl_file.append('            }' + "\n")
        _core_krnl_file.append('        }' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    // Final write.' + "\n")
        _core_krnl_file.append('    if ( (dist_1 <= dist_2) && (dist_1 <= dist_3) ){' + "\n")
        _core_krnl_file.append('        write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                        dist_1,' + "\n")
        _core_krnl_file.append('                        id_1,' + "\n")
        _core_krnl_file.append('                        i_req_output, ' + "\n")
        _core_krnl_file.append('                        i_resp_output);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    else if ( (dist_2 <= dist_3) && (dist_2 <= dist_1) ){' + "\n")
        _core_krnl_file.append('        write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                        dist_2,' + "\n")
        _core_krnl_file.append('                        id_2,' + "\n")
        _core_krnl_file.append('                        i_req_output, ' + "\n")
        _core_krnl_file.append('                        i_resp_output);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    else{' + "\n")
        _core_krnl_file.append('        write_out_mmap( hiermerge_output,' + "\n")
        _core_krnl_file.append('                        dist_3,' + "\n")
        _core_krnl_file.append('                        id_3,' + "\n")
        _core_krnl_file.append('                        i_req_output, ' + "\n")
        _core_krnl_file.append('                        i_resp_output);' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('    #ifndef __SYNTHESIS__' + "\n")
        _core_krnl_file.append('    if (debug_PE_idx == 0)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('        printf("KDEBUG: FINAL Hiermerge, Emptying the input FIFOs now...\\n");' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    #endif' + "\n")
        _core_krnl_file.append('    // Empty the input streams.' + "\n")
        _core_krnl_file.append('    while (stream1_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream1_read_count;' + "\n")
        _core_krnl_file.append('        dist_1 = hiermerge_dist_istream_1.read();' + "\n")
        _core_krnl_file.append('        id_1 = hiermerge_id_istream_1.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    while (stream2_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP/2 max=TOP/2' + "\n")
        _core_krnl_file.append('        ++stream2_read_count;' + "\n")
        _core_krnl_file.append('        dist_2 = hiermerge_dist_istream_2.read();' + "\n")
        _core_krnl_file.append('        id_2 = hiermerge_id_istream_2.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('    while (stream3_read_count < TOP)' + "\n")
        _core_krnl_file.append('    {' + "\n")
        _core_krnl_file.append('    #pragma HLS loop_tripcount min=TOP max=TOP' + "\n")
        _core_krnl_file.append('        ++stream3_read_count;' + "\n")
        _core_krnl_file.append('        dist_3 = hiermerge_dist_istream_3.read();' + "\n")
        _core_krnl_file.append('        id_3 = hiermerge_id_istream_3.read();' + "\n")
        _core_krnl_file.append('    }' + "\n")
        _core_krnl_file.append('}' + "\n")
        _core_krnl_file.append('' + "\n")
        _core_krnl_file.append('' + "\n")






def Generate_singlePE_output_function(_core_krnl_file, _knn_config):
    _core_krnl_file.append('void krnl_singlePE_Write_Outputs(' + "\n")
    _core_krnl_file.append('    tapa::istream<DATA_TYPE> &in_dist0,' + "\n")
    _core_krnl_file.append('    tapa::istream<int> &in_id0,' + "\n")
    _core_krnl_file.append('    tapa::async_mmap<INT32>& output_knn' + "\n")
    _core_krnl_file.append(') {' + "\n")
    _core_krnl_file.append('    DATA_TYPE cur_dist;' + "\n")
    _core_krnl_file.append('    int cur_id;' + "\n")
    _core_krnl_file.append('    int i_req_output = 2*TOP-1; ' + "\n")
    _core_krnl_file.append('    int i_resp_output = 2*TOP-1;' + "\n")
    _core_krnl_file.append('    for (int i = 0; i < TOP; ++i ) {' + "\n")
    _core_krnl_file.append('        #pragma HLS pipeline II=1' + "\n")
    _core_krnl_file.append('        cur_dist = in_dist0.read();' + "\n")
    _core_krnl_file.append('        cur_id = in_id0.read();' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        write_out_mmap(' + "\n")
    _core_krnl_file.append('            output_knn,' + "\n")
    _core_krnl_file.append('            cur_dist,' + "\n")
    _core_krnl_file.append('            cur_id,' + "\n")
    _core_krnl_file.append('            i_req_output,' + "\n")
    _core_krnl_file.append('            i_resp_output' + "\n")
    _core_krnl_file.append('        );' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('    }' + "\n")
    _core_krnl_file.append('}' + "\n")



def Generate_KNN_TopLevel(_core_krnl_file, _num_PE, _knn_config):
    if (_knn_config.using_Ltypes == 0):
        LTYPE_OR_DTYPE = "LOCAL_DIST_DTYPE"
    if (_knn_config.using_Ltypes == 1):
        LTYPE_OR_DTYPE = "DATA_TYPE"

    groupings_per_globalsort_layer = []
    num_arr_per_globalsort_layer = []
    cur_layer_num_arr = _num_PE
    num_arr_per_globalsort_layer.append(cur_layer_num_arr)
    using_interPE_merge = 0

    if (_num_PE > 1):
        tmp_arr_ctr = _num_PE
        using_interPE_merge = 1
        while (cur_layer_num_arr > 1):
            cur_layer_groupings = []
            cur_layer_num_arr = 0

            while (tmp_arr_ctr > 0):
                if (tmp_arr_ctr != 3):
                    # Do a 2-1 merge
                    cur_layer_num_arr += 1
                    tmp_arr_ctr -= 2
                    cur_layer_groupings.append(2)
                else:
                    # Do a 3-1 merge
                    cur_layer_num_arr += 1
                    tmp_arr_ctr -= 3
                    cur_layer_groupings.append(3)

            tmp_arr_ctr = cur_layer_num_arr
            groupings_per_globalsort_layer.append(cur_layer_groupings)
            if (cur_layer_num_arr > 1):
                num_arr_per_globalsort_layer.append(cur_layer_num_arr)

    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('void Knn(' + "\n")
    for PE_idx in range(_num_PE):
        _core_krnl_file.append('    tapa::mmap<INTERFACE_WIDTH> in_' + str(int(PE_idx)) + ',' + "\n")
    _core_krnl_file.append('    tapa::mmap<INT32> final_out' + "\n")
    _core_krnl_file.append(') {' + "\n")
    _core_krnl_file.append('' + "\n")
    
    _core_krnl_file.append('    // Streams, for the global merge:' + "\n")
    for (layer_idx, num_arr) in enumerate(num_arr_per_globalsort_layer):
        if (layer_idx == 0):
            _core_krnl_file.append('    tapa::streams<DATA_TYPE, ' + str(num_arr) + ', TOP> L' + str(layer_idx) + '_out_dist;' + "\n")
            _core_krnl_file.append('    tapa::streams<int,       ' + str(num_arr) + ', TOP> L' + str(layer_idx) + '_out_id;' + "\n")
        else:
            _core_krnl_file.append('    tapa::streams<DATA_TYPE, ' + str(num_arr) + ', 2>   L' + str(layer_idx) + '_out_dist;' + "\n")
            _core_krnl_file.append('    tapa::streams<int,       ' + str(num_arr) + ', 2>   L' + str(layer_idx) + '_out_id;' + "\n")


    ###################################
    #### Generating the load/compute/sort/intraPEmerge
    #### Fully-task-parallel, streaming implementation:
    _core_krnl_file.append('    // Streams, for load->compute->sort:' + "\n")
    _core_krnl_file.append('    tapa::streams<INTERFACE_WIDTH, NUM_PE, 2>           load_to_compute_stream;' + "\n")
    for PE_idx in range(_num_PE):
        if (_knn_config.using_Ltypes == 1):
            _core_krnl_file.append('    tapa::streams<LOCAL_DIST_DTYPE, NUM_SEGMENTS, 2>    compute_to_sort_stream_' + str(PE_idx) + ';' + "\n")
        else:
            _core_krnl_file.append('    tapa::streams<DATA_TYPE, NUM_SEGMENTS, 2>           compute_to_sort_stream_' + str(PE_idx) + ';' + "\n")

    if (_knn_config.using_intra_pe_merge):
        for PE_idx in range(_num_PE):
            _core_krnl_file.append('    tapa::streams<DATA_TYPE, NUM_SEGMENTS*D2L_FACTOR_W, TOP>                 sort_to_hiermerge_dist_stream_' + str(PE_idx) + ';' + "\n")
            _core_krnl_file.append('    tapa::streams<int, NUM_SEGMENTS*D2L_FACTOR_W, TOP>                       sort_to_hiermerge_id_stream_' + str(PE_idx) + ';' + "\n")
    _core_krnl_file.append('' + "\n")

    if (_knn_config.using_intra_pe_merge):
        for PE_idx in range(_num_PE):
            _core_krnl_file.append('    HIERMERGE_STREAM_DECLS(' + str(PE_idx) + ')' + "\n")
        _core_krnl_file.append('' + "\n")

    _core_krnl_file.append('' + "\n")

    ### End of stream declarations.
    ### Begin invocations

    _core_krnl_file.append('    tapa::task()' + "\n")
    for PE_idx in range(_num_PE):
        _core_krnl_file.append('        .invoke( load_KNN, ' + str(PE_idx) + ', in_' + str(PE_idx) + ', load_to_compute_stream[' + str(PE_idx) + '])' + "\n")
    _core_krnl_file.append('' + "\n")
    for PE_idx in range(_num_PE):
        _core_krnl_file.append('        .invoke( compute_KNN, ' + str(PE_idx) + ', NUM_SP_PTS_PER_KRNL_PADDED*' + str(PE_idx) +
                                        ', load_to_compute_stream[' + str(PE_idx) + ' ], compute_to_sort_stream_' + str(PE_idx) + '  )' + "\n")
    _core_krnl_file.append('' + "\n")
    for PE_idx in range(_num_PE):
        _core_krnl_file.append('        INVOKE_PPS_UNITS_FOR_PE(' + str(PE_idx) + ')' + "\n")

    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        ////////// MERGING LOGIC' + "\n")
    _core_krnl_file.append('' + "\n")

    if (_knn_config.using_intra_pe_merge):
        for PE_idx in range(_num_PE):
            _core_krnl_file.append('        INVOKE_HIERMERGE_UNITS_FOR_PE(' + str(PE_idx) + ')' + "\n")

    ###################################
    

    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('' + "\n")
    _core_krnl_file.append('        // INTER-PE HIERMERGE:' + "\n")
    ### Generating the inter-PE hierarchical merge:
    if (using_interPE_merge):
        for layer_idx in range(len(num_arr_per_globalsort_layer)):
            _core_krnl_file.append('' + "\n")
            src_arr_idx = 0
            dst_arr_idx = 0
            if (layer_idx == len(num_arr_per_globalsort_layer) - 1):
                is_final_layer = 1
                FINAL_string = "_FINAL"
            else:
                is_final_layer = 0
                FINAL_string = ""

            for grouping in groupings_per_globalsort_layer[layer_idx]:
                if (grouping == 3):
                    _core_krnl_file.append('        .invoke( merge_trio_streams' + FINAL_string + ', ')
                    _core_krnl_file.append('-1, ' + str(src_arr_idx) + ', ' + str(layer_idx-1) + ', ')
                    _core_krnl_file.append('L' + str(layer_idx) + '_out_dist[' + str(src_arr_idx+0) + '], L' + str(layer_idx) + '_out_id[' + str(src_arr_idx+0) + '], ')
                    _core_krnl_file.append('L' + str(layer_idx) + '_out_dist[' + str(src_arr_idx+1) + '], L' + str(layer_idx) + '_out_id[' + str(src_arr_idx+1) + '], ')
                    _core_krnl_file.append('L' + str(layer_idx) + '_out_dist[' + str(src_arr_idx+2) + '], L' + str(layer_idx) + '_out_id[' + str(src_arr_idx+2) + '], ')
                    if (is_final_layer):
                        _core_krnl_file.append('final_out);' + "\n")
                    else:
                        _core_krnl_file.append('L' + str(layer_idx+1) + '_out_dist[' + str(dst_arr_idx) + '], L' + str(layer_idx+1) + '_out_id[' + str(dst_arr_idx) + '] )' + "\n")
                elif (grouping == 2):
                    _core_krnl_file.append('        .invoke( merge_dual_streams' + FINAL_string + ', ')
                    _core_krnl_file.append('-1, ' + str(src_arr_idx) + ', ' + str(layer_idx-1) + ', ')
                    _core_krnl_file.append('L' + str(layer_idx) + '_out_dist[' + str(src_arr_idx+0) + '], L' + str(layer_idx) + '_out_id[' + str(src_arr_idx+0) + '], ')
                    _core_krnl_file.append('L' + str(layer_idx) + '_out_dist[' + str(src_arr_idx+1) + '], L' + str(layer_idx) + '_out_id[' + str(src_arr_idx+1) + '], ')
                    if (is_final_layer):
                        _core_krnl_file.append('final_out);' + "\n")
                    else:
                        _core_krnl_file.append('L' + str(layer_idx+1) + '_out_dist[' + str(dst_arr_idx) + '], L' + str(layer_idx+1) + '_out_id[' + str(dst_arr_idx) + '] )' + "\n")
                else:
                    msg = "SOMETHING HAS GONE WRONG. Grouping = " + str(grouping)
                    print(msg)
                    _core_krnl_file.append(msg)
                    sys.exit(-5)

                src_arr_idx += grouping
                dst_arr_idx += 1


    if (not using_interPE_merge):
        _core_krnl_file.append('        .invoke( krnl_singlePE_Write_Outputs,  L0_out_dist[0],  L0_out_id[0],' + "\n")
        _core_krnl_file.append('                                               final_out);' + "\n")

    _core_krnl_file.append('}' + "\n")

