## Introduction

CHIP-KNN is the framework for a configurable and high-performance K-Nearest Neighbors accelerator on cloud FPGAs. It automatically generates bandwidth-optimized KNN accelerator on cloud FPGA platforms.

If you use CHIP-KNN in your research, please cite our ICFPT'2020 paper:
> Alec Lu, Zhenman Fang, Nazanin Farahpour, Lesley Shannon. CHIP-KNN: A Configurable and High-Performance K-Nearest Neighbors Accelerator on Cloud FPGAs. IEEE International Conference on Field-Programmable Technology (ICFPT 2020). Virtual Conference, December 2020.

## Download CHIP-KNN

	git clone https://github.com/SFU-HiAccel/CHIP-KNN.git

## Setup Requirements

1. **Evaluated hardware platforms:**
    * **Host CPU:**
      * 64-bit Ubuntu 16.04.6 LTS
    * **Cloud FPGA:**
      * Xilinx Alveo U200 - DDR4-based FPGA
      * Xilinx Alveo U280 - HBM2-based FPGA

2. **Software tools:**
    * **HLS tool:**
      * Vitis 2019.2
      * Xilinx Runtime(XRT) 2019.2

## Accelerate KNN Algorithm using CHIP-KNN

Assuming you are in the project home directory (`$CHIP-KNN_HOME`) of the checked out CHIP-KNN github repo.

1. Configure CHIP-KNN
    * Change directory to `<$CHIP-KNN_HOME>/scripts/`
    * Update KNN parameters:
      * **N** - # of points in search space,
      * **D** - # of data dimension,
      * **Dist** - distance metric,
      * **K** - # of nearest neighbors
    * Update FPGA platform specifications:
      * **FPGA_part_name** - FPGA part name used during Vivado_HLS synthesis
      * **num_SLR** - # of Super-Logic-Regions (SLRs) on the FPGA
      * **SLR_resource** - available resources on each SLR (FPGA dies)
      * **memory** - off-chip memory type on the FPGA
      * **num_memory_banks** - # of available memory banks on the FPGA
    * Update advanced design configurations (***optional**):
      * **singlePE_template_config** - list of single PE configurations based on port_width and buffer_size to explore 
      * **resource_limit** - upper limit scale for the FPGA resource utilization
      * **kernel_frequency** - frequency constraint when building hw kernels
    
2. Explore Single-PE Performance
    * Change directory to `<$CHIP-KNN_HOME>/scripts/`
    * Run: `python singlePE_design_exploration.py`
    * Output: `singlePE_perf_results.log`
      * Each line represents the configuration, bandwidth utilization, and resource usage of a single PE design
    
3. Generate Multi-PE Design
    * Change directory to `<$CHIP-KNN_HOME>/scripts/`
    * Run: `python multiPE_design_exploration.py`
    * Output: `<$CHIP-KNN_HOME>/scripts/gen_test`
      * SW Host Code: `.../gen_test/src/host.cpp`
      * HW Kernel Code: `.../gen_test/src/krnl_*`
      * Connectivity: `.../gen_test/src/knn.ini`

4. Build CHIP-KNN Design
    * Change directory to `<$CHIP-KNN_HOME>/scripts/gen_test`
    * Run: `make build TARGET=hw DEVICE=<FPGA platform>`
    * Output: 
      * host code: `knn`
      * kernel code: `build_dir.hw.<FPGA platform>/knn.xclbin`

5. Run CHIP-KNN 
    * Change directory to `<$CHIP-KNN_HOME>/scripts/gen_test`
    * Run: `make check TARGET=hw DEVICE=<FPGA platform>` or `./knn knn.xclbin`

Now you have completed the flow of the framework. Hack the code and have fun!

## Contacts

Still have further questions about CHIP-KNN? Please contact:

* **Alec Lu**, PhD Student

* HiAccel Lab & RCL Lab, Simon Fraser University (SFU)

* Email: alec_lu@sfu.ca 

* Website: http://www.sfu.ca/~fla30/
