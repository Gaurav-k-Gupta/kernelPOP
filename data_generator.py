import argparse
import pandas as pd
from sklearn.datasets import make_circles
import numpy as np

def generate_data(n, d, output_file):
    # Generating non-linearly separable data
    X, _ = make_circles(n_samples=n, factor=0.4, noise=0.05, random_state=42)

    # Pad with zeros if d > 2
    if d > 2:
        padding = np.zeros((n, d - 2))
        X = np.hstack((X, padding))

    df = pd.DataFrame(X)
    df.to_csv(output_file, index=False, header=False)
    print(f"Generated {n} points with {d} features and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000, help='Number of points')
    parser.add_argument('-d', type=int, default=2, help='Number of features')
    parser.add_argument('-o', type=str, default='data.csv', help='Output file')
    args = parser.parse_args()
    generate_data(args.n, args.d, args.o)