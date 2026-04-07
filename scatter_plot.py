import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Usage: python scatter_plot.py <data.csv> <clusters.csv> <output.png>")
    sys.exit(1)

data = pd.read_csv(sys.argv[1], header=None)
clusters = pd.read_csv(sys.argv[2], header=None)

plt.figure(figsize=(8, 6))
plt.scatter(data[0], data[1], c=clusters[0], cmap='viridis', marker='o', edgecolors='k')
plt.title("Kernel K-Means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

output_img = sys.argv[3] if len(sys.argv) > 3 else 'plot.png'
plt.savefig(output_img)
print(f"Plot saved to {output_img}")