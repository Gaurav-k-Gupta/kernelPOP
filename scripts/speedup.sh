#!/bin/bash
set -e

# 1. Array of dataset paths (Update these with your actual 4 paths)
DATASETS=(
    "./Datasets/acoustic.csv"
    "./Datasets/cifar10.csv"
    "./Datasets/letter.csv"
    "./Datasets/mnist.csv"
)

# Values of k to test based on the paper's figures
K_VALS=(10 50 100)

# CSV to store the parsed results
RESULTS_FILE="benchmark_results.csv"
echo "Dataset,K,CPU_Time,GPU_Normal_Time,Popcorn_Dist_Time" > "$RESULTS_FILE"

echo "=========================================================="
echo "          STARTING KERNEL K-MEANS BENCHMARKS              "
echo "=========================================================="

for DATA_FILE in "${DATASETS[@]}"; do
    # Extract the base name of the dataset for the plot labels (e.g., "acoustic")
    DATASET_NAME=$(basename "$DATA_FILE" .csv)
    
    # Calculate N and D on the fly
    N=$(wc -l < "$DATA_FILE")
    D=$(awk -F',' 'NR==1 {print NF}' "$DATA_FILE")
    
    echo "-> Processing $DATASET_NAME (N=$N, D=$D)"

    for K in "${K_VALS[@]}"; do
        echo "   Running k=$K..."

        # --- 1. Run CPU Normal ---
        OUT_CPU=$(./cpu_normal $N $D $K "$DATA_FILE")
        TIME_CPU=$(echo "$OUT_CPU" | awk '/CPU Normal Time:/ {print $4}')

        # --- 2. Run GPU Normal (Baseline) ---
        OUT_GPU=$(./gpu_normal $N $D $K "$DATA_FILE")
        TIME_GPU=$(echo "$OUT_GPU" | awk '/GPU Normal Time:/ {print $4}')

        # --- 3. Run GPU Matrix (Popcorn) ---
        OUT_MAT=$(./gpu_matrix $N $D $K 50 1 2 gemm "$DATA_FILE")
        
        # Extract the last line which contains "Gram_ms Total_ms"
        LAST_LINE=$(echo "$OUT_MAT" | tail -n 1)
        GRAM_MS=$(echo "$LAST_LINE" | awk '{print $1}')
        TOTAL_MS=$(echo "$LAST_LINE" | awk '{print $2}')

        # Calculate Popcorn Distances Time in seconds. 
        # (Total GPU time - Gram Matrix time) / 1000
        POPCORN_DIST_SEC=$(awk "BEGIN {printf \"%.6f\", ($TOTAL_MS - $GRAM_MS) / 1000}")
        # If you strictly want to use JUST the Gram matrix time, uncomment the line below instead:
        # POPCORN_DIST_SEC=$(echo "scale=6; $GRAM_MS / 1000" | bc)

        # Append to results CSV
        echo "$DATASET_NAME,$K,$TIME_CPU,$TIME_GPU,$POPCORN_DIST_SEC" >> "$RESULTS_FILE"
    done
done

echo "=========================================================="
echo "          BENCHMARKS COMPLETE. GENERATING PLOTS.          "
echo "=========================================================="

# 2. Generate the Python script to plot the results
cat << 'EOF' > plot_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the benchmark data
df = pd.read_csv('benchmark_results.csv')
datasets = df['Dataset'].unique()
k_vals = sorted(df['K'].unique())

x = np.arange(len(datasets))
width = 0.25  # Width of the bars

# ---------------------------------------------------------
# Figure 3: Speedup of Baseline CUDA over CPU
# ---------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))
colors_fig3 = ['#FFE4C4', '#FFA500', '#FF8C00'] # Light orange, Orange, Dark orange

for i, k in enumerate(k_vals):
    subset = df[df['K'] == k]
    # Speedup = CPU Time / Baseline GPU Time
    speedups = subset['CPU_Time'] / subset['GPU_Normal_Time']
    
    # Offset the bars to group them side-by-side
    offset = (i - 1) * width
    ax3.bar(x + offset, speedups, width, label=f'k={k}', color=colors_fig3[i], edgecolor='black', zorder=3)

ax3.set_ylabel('Speedup (x)')
ax3.set_xlabel('Dataset')
ax3.set_title('Speedup of Baseline CUDA Implementation Over CPU Implementation')
ax3.set_xticks(x)
ax3.set_xticklabels(datasets)
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig('Figure_3.png', dpi=300)
print("Saved Figure_3.png")

# ---------------------------------------------------------
# Figure 4: Speedup of Popcorn Distances over Baseline CUDA
# ---------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 6))
colors_fig4 = ['#87CEEB', '#4682B4', '#000080'] # Light blue, Steel blue, Navy blue

for i, k in enumerate(k_vals):
    subset = df[df['K'] == k]
    # Speedup = Baseline GPU Time / Popcorn Pairwise Distances Time
    speedups = subset['GPU_Normal_Time'] / subset['Popcorn_Dist_Time']
    
    offset = (i - 1) * width
    ax4.bar(x + offset, speedups, width, label=f'k={k}', color=colors_fig4[i], edgecolor='black', zorder=3)

ax4.set_ylabel('Speedup (x)')
ax4.set_xlabel('Dataset')
ax4.set_title('Speedup of Popcorn Distances Algorithm over Baseline CUDA Implementation')
ax4.set_xticks(x)
ax4.set_xticklabels(datasets)
ax4.legend()
ax4.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig('Figure_4.png', dpi=300)
print("Saved Figure_4.png")

EOF

# 3. Execute the Python plotting script
python3 plot_figures.py

echo "All done! Check your directory for Figure_3.png and Figure_4.png."