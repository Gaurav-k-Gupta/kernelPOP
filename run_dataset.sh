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
K=5

echo "=> Detected Dataset: $DATA_FILE with N=$N and D=$D"
echo "=> Setting up python environment..."
# pip install -r requirements.txt -q

echo "=> Compiling CUDA file..."

# CRITICAL: Added -lcublas (and -lcusparse if your popcorn uses it)
nvcc matrix_centric_kernel_k_means_gpu.cu -o gpu_matrix -O3 -Wno-deprecated-gpu-targets -lcublas -lcusparse



    G=50
    OUT_GPU_M=$(./gpu_matrix $N $D $K $G 1 2 GEMM "$DATA_FILE")
    
    # Extract execution time
    TIME_GPU_M=$(echo "$OUT_GPU_M" | awk '/Gram matrix/ { if ($6 ~ /ms/) printf "%.6f", $5/1000; else print $5 }')
    
    t1=$(echo "$OUT_GPU_M" | tail -n 1 | awk '{print $1}')
    t2=$(echo "$OUT_GPU_M" | tail -n 1 | awk '{print $2}')
    
    printf "$t1 $t2 \n"

    # Rename outputs to avoid overwriting
    CSV_FILE="out_gpu_matrix_g${G}.csv"
    PLOT_FILE="gpu_matrix_plot_g${G}.png"
    mv out_gpu_gemm.csv $CSV_FILE
    
    printf "| %-19s | %-5s | %-18s | %-18s |\n" "GPU Matrix-Centric" "$G" "$TIME_GPU_M" "$PLOT_FILE"
    
    # Generate scatter plot in the background (&) to save time, using $DATA_FILE
    # python scatter_plot.py "$DATA_FILE" $CSV_FILE $PLOT_FILE &
# done

# Wait for all background plotting jobs to finish
wait 

echo "========================================================================="
echo ""
echo "Done!"