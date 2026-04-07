#!/bin/bash
set -e

N=1000
D=10
K=2

echo "=> Setting up python environment and generating data..."
pip install -r requirements.txt -q
python data_generator.py -n $N -d $D -o data.csv

echo "=> Compiling C and CUDA files..."
gcc kernel_k_means_cpu.c -o cpu_normal -O3 -lm
nvcc kernel_k_means_gpu.cu -o gpu_normal -O3 -Wno-deprecated-gpu-targets
gcc matrix_centric_kernel_k_means_cpu.c -o cpu_matrix -O3 -lm

# CRITICAL: Added -lcublas to link the cuBLAS library!
nvcc matrix_centric_kernel_k_means_gpu.cu -o gpu_matrix -O3 -Wno-deprecated-gpu-targets -lcublas

echo "=> Executing algorithms with N=$N, D=$D, K=$K..."
OUT_CPU_N=$(./cpu_normal $N $D $K)
OUT_GPU_N=$(./gpu_normal $N $D $K)
OUT_CPU_M=$(./cpu_matrix $N $D $K)
OUT_GPU_M=$(./gpu_matrix $N $D $K)

echo "=> Generating Scatter Plot for Matrix-Centric GPU results..."
python scatter_plot.py data.csv out_gpu_matrix.csv gpu_matrix_plot.png

echo ""
echo "=========================================================="
echo "                   PERFORMANCE TABLE                      "
echo "=========================================================="
echo "| Implementation               | Execution Time (s)      |"
echo "|------------------------------|-------------------------|"
printf "| %-28s | %-23s |\n" "CPU Normal K-Means" "$(echo $OUT_CPU_N | awk '{print $4}')"
printf "| %-28s | %-23s |\n" "GPU Normal K-Means" "$(echo $OUT_GPU_N | awk '{print $4}')"
printf "| %-28s | %-23s |\n" "CPU Matrix-Centric K-Means" "$(echo $OUT_CPU_M | awk '{print $4}')"
printf "| %-28s | %-23s |\n" "GPU Matrix-Centric K-Means" "$(echo $OUT_GPU_M | awk '{print $4}')"
echo "=========================================================="
echo ""
echo "Done! Check gpu_matrix_plot.png for visual results."