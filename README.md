## Introduction

CHIP-KNN is the framework for a configurable and high-performance K-Nearest Neighbors accelerator on cloud FPGAs. It automatically generates bandwidth-optimized KNN accelerator on cloud FPGA platforms.

There are two versions:
  * CHIP-KNN (v1), which is a double-buffered design that is capable of saturating DDR bandwidth.
  * CHIP-KNNv2, which is a streaming-based and frequency-optimized extension of v1, significantly improving performance on HBM-based FPGAs.

If you use CHIP-KNNv2 in your research, please cite our TRETS'2023 paper:
> Kenneth Liu, Alec Lu, Kartik Samtani, Zhenman Fang, Licheng Guo. CHIP-KNNv2: A Configurable and High-Performance K-Nearest Neighbors Accelerator on HBM-based FPGAs. ACM Transactions on Reconfigurable Technology and Systems, Vol. 16, No. 4, December 2023. DOI: 10.1145/3616873

If you use CHIP-KNN (v1) in your research, please cite our ICFPT'2020 paper:
> Alec Lu, Zhenman Fang, Nazanin Farahpour, Lesley Shannon. CHIP-KNN: A Configurable and High-Performance K-Nearest Neighbors Accelerator on Cloud FPGAs. IEEE International Conference on Field-Programmable Technology (ICFPT 2020). Virtual Conference, December 2020.


## Setup Requirements

1. **Evaluated hardware platforms:**
    * **Host OS:**
      * 64-bit Ubuntu 16.04.6 LTS
    * **Cloud FPGA:**
      * Xilinx Alveo U280 - HBM2-based FPGA

2. **Software tools:**
    * **HLS tool:**
      * Vitis (Tested with v2020.2)
      * Xilinx Runtime(XRT) (Tested with v2020.2)
      * TAPA (Tested with version 0.0.20220807.1)
        * https://github.com/UCLA-VAST/tapa/
        * https://tapa.readthedocs.io/en/release/overview/overview.html

## Usage: Accelerating the KNN Algorithm using CHIP-KNN

Assuming you are in the directory (`$CHIP-KNN_HOME/scripts`) of the cloned CHIP-KNN github repo.

1. Configure CHIP-KNN
    * In config.py, change the KNN parameters as desired:
      * **N** - # of points in search space,
      * **D** - data dimension,
      * **Dist** - distance metric,
      * **K** - # of nearest neighbors to return
    * Also in config.py, Update FPGA platform specifications:
      * **FPGA_part_name** - FPGA part name used during Vivado_HLS synthesis
      * **SLR_resource** - available resources on each SLR (FPGA dies)
      * **memory_type** - off-chip memory type on the FPGA
      * **num_mem_banks** - # of available memory banks on the FPGA
    * Update advanced design configurations (***optional***):
      * **resource_limit** - targeted upper limit of FPGA resource utilization, when replicating Processing Elements (PEs)
    
2. Explore Single-PE Resource Utilization
    * Run: `python singlePE_design_exploration.py`
    * Outputs: 
      * `singlePE_perf_results.log`
        * This contains the configuration and resource usage of a single PE design. Used by the multi-PE exploration script.
      * `gen_singlePE_design/`
        * This contains the code for the single-PE design.
    * NOTE: If you experience compilation issues, you may need to modify the include directories in the Makefile. The Makefile is auto-generated in scripts/supporting_code_generation.py, in the function named "Generate_Makefile". Please modify the lines defining "CXXFLAGS", to include your Vitis_HLS install directory.

3. Generate Multi-PE Design
    * Run: `python multiPE_design_exploration.py`
    * Output: `gen_multiPE_design/`
      * SW Host Code: `.../gen_test/src/host.cpp`
      * HW Kernel Code: `.../gen_test/src/krnl_*`
      * Connectivity: `.../gen_test/src/knn.ini`

4. Build CHIP-KNN Design
    * Change directory to `<$CHIP-KNN_HOME>/scripts/gen_multiPE_design/`
    * Run: `make build TARGET=hw DEVICE=<FPGA platform>`
    * Output: 
      * host code: `knn`
      * kernel code: `build_dir.hw.<FPGA platform>/knn.xclbin`

5. Run CHIP-KNN 
    * Change directory to `<$CHIP-KNN_HOME>/scripts/gen_multiPE_design/`
    * Run: `make check TARGET=hw DEVICE=<FPGA platform>` or `./knn knn.xclbin`

Now you have completed the flow of the framework. Hack the code and have fun!

## Contacts

Still have further questions about CHIP-KNNv2? Please contact:

* **Kenneth (Kenny) Liu**, MASc Student

* HiAccel Lab, Simon Fraser University (SFU)

* Email: `ksl24 [at] sfu [dot] ca`

* Website: http://www.sfu.ca/~ksl24/
