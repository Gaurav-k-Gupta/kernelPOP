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

2. **Run the Master Script**
    Give execution rights to the provided bash script and run it. The script handles everything end-to-end:
    ```bash
    chmod +x script.sh
    ./script.sh
    ```

3. **What the Script Does:**

    Calls data_generator.py to create a data.csv file with 1000 points arranged in non-linear concentric circles.

    Compiles kernel_k_means_cpu.c and kernel_k_means_gpu.cu (Baseline implementation).

    Compiles matrix_centric_kernel_k_means_cpu.c and matrix_centric_kernel_k_means_gpu.cu (Proposed implementation).

    Records the exact time taken by the main execution components.

    Formats the data into a markdown terminal table.

    Calls scatter_plot.py to map the points into a generated .png visual confirming convergence.