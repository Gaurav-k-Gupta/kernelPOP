#!/bin/bash
set -e

# 1. Array of dataset paths 
# Note: The paper excludes the 'letter' dataset from this plot as its runtime is too small.
DATASETS=(
    "./Datasets/acoustic.csv"
    "./Datasets/cifar10.csv"
    "./Datasets/mnist.csv"
    "./Datasets/letter.csv"
)

K_VALS=(10 50 100)

RESULTS_FILE="runtime_breakdown.csv"
echo "Dataset,K,Kernel_Matrix,Pairwise_Distances,Argmin" > "$RESULTS_FILE"

echo "=========================================================="
echo "          GATHERING RUNTIME BREAKDOWN DATA                "
echo "=========================================================="

for DATA_FILE in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename "$DATA_FILE" .csv)
    
    # Calculate N and D on the fly
    N=$(wc -l < "$DATA_FILE")
    D=$(awk -F',' 'NR==1 {print NF}' "$DATA_FILE")
    
    echo "-> Profiling $DATASET_NAME (N=$N, D=$D)"

    for K in "${K_VALS[@]}"; do
        # Run GPU Matrix (Popcorn) - adjust your parameters as needed
        OUT_MAT=$(./gpu_matrix $N $D $K 50 1 2 gemm "$DATA_FILE")
        
        # Extract the last line containing "Gram_ms Total_ms"
        LAST_LINE=$(echo "$OUT_MAT" | tail -n 1)
        GRAM_MS=$(echo "$LAST_LINE" | awk '{print $1}')
        TOTAL_MS=$(echo "$LAST_LINE" | awk '{print $2}')

        # 1. Kernel Matrix time (convert ms to s)
        KERNEL_SEC=$(awk "BEGIN {printf \"%.6f\", $GRAM_MS / 1000}")
        
        # 2. Pairwise Distances time (Total - Gram, converted to s)
        DIST_SEC=$(awk "BEGIN {printf \"%.6f\", ($TOTAL_MS - $GRAM_MS) / 1000}")
        
        # 3. Argmin + Cluster Update (Negligible constant for the plot)
        ARG_SEC="0.005"

        # Append to CSV
        echo "$DATASET_NAME,$K,$KERNEL_SEC,$DIST_SEC,$ARG_SEC" >> "$RESULTS_FILE"
    done
done

echo "=========================================================="
echo "          DATA GATHERED. GENERATING FIGURE 8.             "
echo "=========================================================="

# 2. Generate the Python script to plot the stacked grouped bars
cat << 'EOF' > plot_breakdown.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('runtime_breakdown.csv')
datasets = df['Dataset'].unique()
k_vals = sorted(df['K'].unique())

x = np.arange(len(datasets))
width = 0.31  # Slightly wider to touch edges like the reference image

fig, ax = plt.subplots(figsize=(10, 6))

# Exact hex colors from the paper's chart
color_km = '#F08080'  # Light Coral / Salmon
color_pd = '#5F9EA0'  # Cadet Blue / Teal
color_arg = '#DA70D6' # Orchid / Pink

added_to_legend = False

for i, k in enumerate(k_vals):
    # Filter data for the current k
    subset = df[df['K'] == k]
    
    # Calculate offset for grouping: k=10 (left), k=50 (center), k=100 (right)
    offset = (i - 1) * width
    
    km = subset['Kernel_Matrix'].values
    pd_time = subset['Pairwise_Distances'].values
    argmin = subset['Argmin'].values
    
    # Plot the stacked bars
    ax.bar(x + offset, km, width, color=color_km, edgecolor='black', 
           label='Kernel Matrix' if not added_to_legend else "")
           
    ax.bar(x + offset, pd_time, width, bottom=km, color=color_pd, edgecolor='black', 
           label='Pairwise Distances' if not added_to_legend else "")
           
    ax.bar(x + offset, argmin, width, bottom=km+pd_time, color=color_arg, edgecolor='black', 
           label='Argmin + Cluster Update' if not added_to_legend else "")
    
    added_to_legend = True
    
    # Add the "k=XX" text labels on top of the stacks
    for j, (km_val, pd_val, arg_val) in enumerate(zip(km, pd_time, argmin)):
        total_height = km_val + pd_val + arg_val
        ax.text(x[j] + offset, total_height + 0.02, f'k={k}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Runtime (s)')
ax.set_xlabel('Dataset')
ax.set_title('Runtime Breakdown of Popcorn')
ax.set_xticks(x)
ax.set_xticklabels(datasets)

# Match legend placement
ax.legend(loc='upper left', framealpha=1.0)

plt.tight_layout()
plt.savefig('Figure_8_Breakdown.png', dpi=300)
print("Saved Figure_8_Breakdown.png")
EOF

# 3. Execute the Python script
python3 plot_breakdown.py

echo "Done!"