#!/bin/bash
set -e

# Fixed hyperparameters based on your requirements
K=2
GAMMA=50
COEF=1
DEGREE=2

# Define the N and D combinations exactly as they appear in the target plot
# Format: "N D"
COMBINATIONS=(
    "10000 100"
    "5000 100"
    "5000 1000"
    "1000 1000"
    "1000 100"
    "1000 10"
    "5000 100"
)

echo "=> Setting up python environment..."
#pip install -r requirements.txt -q

echo "=> Compiling CUDA file..."
nvcc matrix_centric_kernel_k_means_gpu.cu -o gpu_matrix -O3 -Wno-deprecated-gpu-targets -lcublas -lcusparse

# Create a CSV to store the results for the plotting script
RESULTS_FILE="benchmark_results.csv"
echo "N,D,GEMM_Time,SYRK_Time" > $RESULTS_FILE

echo ""
echo "================================================================="
echo "                GEMM vs SYRK RUNTIME COMPARISON                  "
echo "================================================================="
echo "| N       | D        | GEMM Time (s) | SYRK Time (s) |"
echo "|---------|----------|---------------|---------------|"

# Run the benchmarking loop
for combo in "${COMBINATIONS[@]}"; do
    read -r N D <<< "$combo"
    
    # Generate the dataset for the current N and D
    OUT_GEN=$(python data_generator.py -n $N -d $D -o data.csv)
    
    # Run GEMM
    OUT_GEMM=$(./gpu_matrix $N $D $K $GAMMA $COEF $DEGREE gemm)
    # Extract "Gram matrix" time. If unit is ms, divide by 1000 to get seconds.
    TIME_GEMM=$(echo "$OUT_GEMM" | awk '/Gram matrix/ { if ($6 ~ /ms/) printf "%.6f", $5/1000; else print $5 }')
    
    # Run SYRK
    OUT_SYRK=$(./gpu_matrix $N $D $K $GAMMA $COEF $DEGREE syrk)
    TIME_SYRK=$(echo "$OUT_SYRK" | awk '/Gram matrix/ { if ($6 ~ /ms/) printf "%.6f", $5/1000; else print $5 }')
    
    printf "| %-7s | %-8s | %-13s | %-13s |\n" "$N" "$D" "$TIME_GEMM" "$TIME_SYRK"
    
    # Append to CSV
    echo "$N,$D,$TIME_GEMM,$TIME_SYRK" >> $RESULTS_FILE
done

echo "================================================================="
echo ""
echo "=> Generating plot..."

# Write a Python plotting script on the fly
cat << 'EOF' > plot_benchmark.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    # Read data
    df = pd.read_csv('benchmark_results.csv')

    # Create x-axis labels format: (n=10000, d=100)
    labels = [f"(n={n}, d={d})" for n, d in zip(df['N'], df['D'])]
    gemm_times = pd.to_numeric(df['GEMM_Time']).values
    syrk_times = pd.to_numeric(df['SYRK_Time']).values

    x = np.arange(len(labels))
    width = 0.40  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Match colors from the screenshot (teal and salmon)
    rects1 = ax.bar(x - width/2, gemm_times, width, label='GEMM', color='teal', edgecolor='black')
    rects2 = ax.bar(x + width/2, syrk_times, width, label='SYRK', color='salmon', edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime of Kernel Matrix Computation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Set y-axis to logarithmic scale as seen in the screenshot
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    plt.savefig('runtime_comparison.png')
    print("Success: Plot saved to 'runtime_comparison.png'")

except Exception as e:
    print(f"Error generating plot: {e}")
    sys.exit(1)
EOF

# Ensure plotting libraries are installed
pip install -q pandas matplotlib
python plot_benchmark.py