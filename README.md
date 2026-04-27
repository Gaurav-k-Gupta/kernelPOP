# Matrix-Centric Kernel K-Means

This project evaluates the speedup introduced by framing Kernel K-Means clustering in terms of linear algebra.

## Prerequisites
- GCC Compiler (`gcc`)
- NVIDIA CUDA Toolkit (`nvcc`)
- Python 3.8+

## Setup Instructions

1. **Install Dependencies**
   The project uses Python dependencies to generate data and plot results. Install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Compile**
You can compile the implementations using the following commands. Ensure you link the necessary NVIDIA math libraries for the matrix-centric build:

```bash
# 1. CPU Baseline
gcc kernel_k_means_cpu.c -o cpu_normal -O3 -lm

# 2. GPU Baseline (Custom Kernels)
nvcc kernel_k_means_gpu.cu -o gpu_normal -O3 -Wno-deprecated-gpu-targets

# 3. GPU Matrix-Centric (Popcorn)
nvcc matrix_centric_kernel_k_means_gpu.cu -o gpu_matrix -O3 -Wno-deprecated-gpu-targets -lcublas -lcusparse
```


3. **Run the Master Script**
    Give execution rights to the provided bash script and run it. The script handles everything end-to-end:
    ```bash
    chmod +x script.sh
    ./script.sh data.csv
    ```

4. **What the Script Does:**

    Calls data_generator.py to create a data.csv file with 1000 points arranged in non-linear concentric circles.

    Compiles kernel_k_means_cpu.c and kernel_k_means_gpu.cu (Baseline implementation).

    Compiles matrix_centric_kernel_k_means_cpu.c and matrix_centric_kernel_k_means_gpu.cu (Proposed implementation).

    Records the exact time taken by the main execution components.

    Formats the data into a markdown terminal table.

    Calls scatter_plot.py to map the points into a generated .png visual confirming convergence.


5. **Explore Other Scripts**

    Explore the scripts/ folder to find more scripts to get different results.

    Example:

    ```bash
    chmod +x gemm_vs_syrk.sh
    ./gemm_vs_syrk.sh

    chmod +x ./scripts/plot_runtimes.sh
    ./scripts/plot_runtimes.sh

    chmod +x ./scripts/speedup.sh
    ./scripts/speedup.sh
    ```

6. **Datasets and Results**


    We have used 4 datasets:-
    acoustic , cifar10 , letter , mnist



    Explore folders like Results/ , Runtimes/ , Speedup/ , GEMM_VS_SYRK/
    to find results 


