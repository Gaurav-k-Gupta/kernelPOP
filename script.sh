#!/bin/bash
set -e

# 1. Check if the user provided a data file argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_file.csv>"
    exit 1
fi

DATA_FILE=$1

# 2. Check if the file actually exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: File '$DATA_FILE' not found!"
    exit 1
fi

# 3. Dynamically determine N (rows) and D (columns) from the CSV
# Count the total number of lines in the file to get N
N=$(wc -l < "$DATA_FILE")

# Read the first line and count the number of comma-separated fields to get D
D=$(awk -F',' 'NR==1 {print NF}' "$DATA_FILE")

# Set K
K=2

echo "=> Detected Dataset: $DATA_FILE with N=$N and D=$D"

# Define the hyperparameters you want to test
GAMMAS=(0.01 0.1 1.0 5.0 10.0 50.0)

echo "=> Setting up python environment..."
pip install -r requirements.txt -q

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
# Appended $DATA_FILE to the arguments
OUT_CPU_N=$(./cpu_normal $N $D $K "$DATA_FILE")
OUT_GPU_N=$(./gpu_normal $N $D $K "$DATA_FILE")

# We assume your C code prints the time in the 4th column, e.g., "CPU Normal Time: 1.23 seconds"
TIME_CPU_N=$(echo $OUT_CPU_N | awk '{print $4}')
TIME_GPU_N=$(echo $OUT_GPU_N | awk '{print $4}')

printf "| %-19s | %-5s | %-18s | %-18s |\n" "CPU Normal Baseline" "N/A" "$TIME_CPU_N" "cpu_normal_plot.png"
printf "| %-19s | %-5s | %-18s | %-18s |\n" "GPU Normal Baseline" "N/A" "$TIME_GPU_N" "gpu_normal_plot.png"

# Passed $DATA_FILE instead of hardcoded 'data.csv'
python scatter_plot.py "$DATA_FILE" out_cpu_normal.csv cpu_normal_plot.png &
python scatter_plot.py "$DATA_FILE" out_gpu_normal.csv gpu_normal_plot.png &

# 2. Run the tuning loop for the Matrix-Centric (Popcorn) implementation
for G in "${GAMMAS[@]}"; do
    # Execute with N D K GAMMA COEF DEGREE METHOD FILENAME
    OUT_GPU_M=$(./gpu_matrix $N $D $K $G 1 2 GEMM "$DATA_FILE")
    
    # Extract execution time
    TIME_GPU_M=$(echo "$OUT_GPU_M" | awk '/Gram matrix/ { if ($6 ~ /ms/) printf "%.6f", $5/1000; else print $5 }')
    
    
    # Rename outputs to avoid overwriting
    CSV_FILE="out_gpu_matrix_g${G}.csv"
    PLOT_FILE="gpu_matrix_plot_g${G}.png"
    mv out_gpu_gemm.csv $CSV_FILE
    
    printf "| %-19s | %-5s | %-18s | %-18s |\n" "GPU Matrix-Centric" "$G" "$TIME_GPU_M" "$PLOT_FILE"
    
    # Generate scatter plot in the background (&) to save time, using $DATA_FILE
    python scatter_plot.py "$DATA_FILE" $CSV_FILE $PLOT_FILE &
done

# Wait for all background plotting jobs to finish
wait 

echo "========================================================================="
echo ""
echo "Done! Check the generated PNG files to see how Gamma affects the clustering boundaries."