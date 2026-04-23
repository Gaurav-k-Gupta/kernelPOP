#!/bin/bash
set -e

N=10000
D=2
K=2

# Define the hyperparameters you want to test
# For RBF, we are tuning Gamma. You can expand this to include Coef/Degree for Polynomial.
GAMMAS=(0.01 0.1 1.0 5.0 10.0 50.0)

echo "=> Setting up python environment and generating data..."
pip install -r requirements.txt -q
python data_generator.py -n $N -d $D -o data.csv

echo "=> Compiling C and CUDA files..."
gcc kernel_k_means_cpu.c -o cpu_normal -O3 -lm
nvcc kernel_k_means_gpu.cu -o gpu_normal -O3 -Wno-deprecated-gpu-targets
gcc matrix_centric_kernel_k_means_cpu.c -o cpu_matrix -O3 -lm

# CRITICAL: Added -lcublas (and -lcusparse if your popcorn uses it)
nvcc matrix_centric_kernel_k_means_gpu.cu -o gpu_matrix -O3 -Wno-deprecated-gpu-targets -lcublas -lcusparse

echo ""
echo "========================================================================="
echo "                   HYPERPARAMETER TUNING TABLE                           "
echo "========================================================================="
echo "| Implementation      | Gamma | Execution Time (s) | Output Plot        |"
echo "|---------------------|-------|--------------------|--------------------|"

# 1. Run the Baseline implementations once (for comparison)
OUT_CPU_N=$(./cpu_normal $N $D $K)
OUT_GPU_N=$(./gpu_normal $N $D $K)

# We assume your C code prints the time in the 4th column, e.g., "CPU Normal Time: 1.23 seconds"
TIME_CPU_N=$(echo $OUT_CPU_N | awk '{print $4}')
TIME_GPU_N=$(echo $OUT_GPU_N | awk '{print $4}')

printf "| %-19s | %-5s | %-18s | %-18s |\n" "CPU Normal Baseline" "N/A" "$TIME_CPU_N" "cpu_normal_plot.png"
printf "| %-19s | %-5s | %-18s | %-18s |\n" "GPU Normal Baseline" "N/A" "$TIME_GPU_N" "gpu_normal_plot.png"

python scatter_plot.py data.csv out_cpu_normal.csv cpu_normal_plot.png &
python scatter_plot.py data.csv out_gpu_normal.csv gpu_normal_plot.png &

# 2. Run the tuning loop for the Matrix-Centric (Popcorn) implementation
for G in "${GAMMAS[@]}"; do
    # Execute with N D K GAMMA
    OUT_GPU_M=$(./gpu_matrix $N $D $K $G)
    
    # Extract execution time
    TIME_GPU_M=$(echo $OUT_GPU_M | awk '{print $4}')
    
    # Rename outputs to avoid overwriting
    CSV_FILE="out_gpu_matrix_g${G}.csv"
    PLOT_FILE="gpu_matrix_plot_g${G}.png"
    mv out_gpu_matrix.csv $CSV_FILE
    
    printf "| %-19s | %-5s | %-18s | %-18s |\n" "GPU Matrix-Centric" "$G" "$TIME_GPU_M" "$PLOT_FILE"
    
    # Generate scatter plot in the background (&) to save time
    python scatter_plot.py data.csv $CSV_FILE $PLOT_FILE &
done

# Wait for all background plotting jobs to finish
wait 

echo "========================================================================="
echo ""
echo "Done! Check the generated PNG files to see how Gamma affects the clustering boundaries."