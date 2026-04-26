#!/usr/bin/env python3
"""
benchmark_gram.py
=================
Compiles kernel_kmeans_gpu.cu, generates synthetic data, runs the binary
with both GEMM and SYRK gram-matrix methods across a range of N values,
then plots a comparison of:
  1. Gram-matrix computation time (ms)
  2. Total GPU pipeline time (s)

Usage
-----
    python benchmark_gram.py [--N 500 1000 2000 4000] [--D 16] [--K 3]
                             [--gamma 5.0] [--iters 3]

Requirements
------------
    nvcc, cublas, matplotlib, numpy
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Benchmark GEMM vs SYRK gram matrix in kernel k-means")
parser.add_argument("--N",     nargs="+", type=int,
                    default=[500, 1000, 2000, 4000],
                    help="List of dataset sizes to benchmark")
parser.add_argument("--D",     type=int, default=16,  help="Feature dimension")
parser.add_argument("--K",     type=int, default=3,   help="Number of clusters")
parser.add_argument("--gamma", type=float, default=5.0, help="RBF gamma")
parser.add_argument("--iters", type=int, default=3,
                    help="Repetitions per (N, method) for averaging")
parser.add_argument("--src",   default="kernel_kmeans_gpu.cu",
                    help="Path to the .cu source file")
parser.add_argument("--bin",   default="./kernel_kmeans_gpu",
                    help="Where to place the compiled binary")
args = parser.parse_args()

SRC      = args.src
BIN      = args.bin
N_LIST   = sorted(args.N)
D        = args.D
K        = args.K
GAMMA    = args.gamma
REPEATS  = args.iters

# ── 1. Compile ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Compiling …")
compile_cmd = [
    "nvcc", "-O3", "-arch=sm_70",   # adjust sm_XX for your GPU
    SRC, "-o", BIN,
    "-lcublas", "-lm",
]
result = subprocess.run(compile_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("Compilation FAILED:")
    print(result.stderr)
    sys.exit(1)
print("Compiled OK →", BIN)


# ── helpers ───────────────────────────────────────────────────────────────────
def generate_data(n: int, d: int, seed: int = 42):
    """Write an N×D CSV of random doubles to data.csv."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    with open("data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for row in X:
            writer.writerow([f"{v:.6f}" for v in row])


def run_once(n: int, method: str) -> dict:
    """
    Run the binary for one (N, method) pair.
    Returns {"gram_ms": float, "total_ms": float}.
    """
    # Clear previous timing log
    if os.path.exists("timing.csv"):
        os.remove("timing.csv")

    cmd = [BIN, str(n), str(D), str(K), str(GAMMA), "1.0", "2.0", method]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Run failed ({method}, N={n}):")
        print(result.stderr)
        return {"gram_ms": float("nan"), "total_ms": float("nan")}

    # Parse timing.csv written by the binary
    try:
        with open("timing.csv") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3 and parts[0].lower() == method.lower():
                    return {
                        "gram_ms":  float(parts[1]),
                        "total_ms": float(parts[2]),
                    }
    except FileNotFoundError:
        pass

    # Fallback: parse stdout
    gram_ms = total_ms = float("nan")
    for line in result.stdout.splitlines():
        if "Gram matrix" in line and "time:" in line:
            gram_ms = float(line.split()[-2])
        if "Total GPU time" in line:
            total_ms = float(line.split()[-2]) * 1000   # s→ms
    return {"gram_ms": gram_ms, "total_ms": total_ms}


# ── 2. Benchmark ──────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Benchmarking  N={N_LIST}  D={D}  K={K}  gamma={GAMMA}  reps={REPEATS}")

results = {m: {"gram_ms": [], "total_ms": []} for m in ("gemm", "syrk")}

for n in N_LIST:
    print(f"\n── N = {n} ──")
    generate_data(n, D)

    for method in ("gemm", "syrk"):
        gram_vals, total_vals = [], []
        for rep in range(REPEATS):
            t = run_once(n, method)
            gram_vals.append(t["gram_ms"])
            total_vals.append(t["total_ms"])
            print(f"  {method.upper():4s}  rep {rep+1}/{REPEATS}"
                  f"  gram={t['gram_ms']:.2f} ms  total={t['total_ms']:.2f} ms")

        results[method]["gram_ms"].append(np.mean(gram_vals))
        results[method]["total_ms"].append(np.mean(total_vals))

print("\n" + "=" * 60)
print("Benchmark complete. Plotting …")


# ── 3. Plot ───────────────────────────────────────────────────────────────────
PALETTE = {
    "gemm": "#2196F3",   # blue
    "syrk": "#FF5722",   # deep orange
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0D1117")
for ax in axes:
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#C9D1D9")
    ax.xaxis.label.set_color("#C9D1D9")
    ax.yaxis.label.set_color("#C9D1D9")
    ax.title.set_color("#E6EDF3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.grid(True, color="#21262D", linewidth=0.8, linestyle="--")

for method, color in PALETTE.items():
    label = method.upper()
    # --- Gram matrix time ---
    gram_vals = results[method]["gram_ms"]
    axes[0].plot(N_LIST, gram_vals, "o-", color=color, linewidth=2,
                 markersize=7, label=label)
    for x, y in zip(N_LIST, gram_vals):
        axes[0].annotate(f"{y:.1f}", xy=(x, y),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8, color=color)

    # --- Total pipeline time ---
    total_vals = results[method]["total_ms"]
    axes[1].plot(N_LIST, total_vals, "s--", color=color, linewidth=2,
                 markersize=7, label=label)
    for x, y in zip(N_LIST, total_vals):
        axes[1].annotate(f"{y:.1f}", xy=(x, y),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8, color=color)

# Speedup annotation on gram subplot
print("\nSpeedups (GEMM time / SYRK time):")
for i, n in enumerate(N_LIST):
    g = results["gemm"]["gram_ms"][i]
    s = results["syrk"]["gram_ms"][i]
    if s > 0 and not np.isnan(s):
        print(f"  N={n:5d}  gram speedup = {g/s:.2f}×")

axes[0].set_title("Gram Matrix Computation Time", fontweight="bold", fontsize=13)
axes[0].set_xlabel("Dataset size  N", fontsize=11)
axes[0].set_ylabel("Time (ms)", fontsize=11)
axes[0].legend(fontsize=11, facecolor="#21262D", labelcolor="#C9D1D9",
               edgecolor="#30363D")

axes[1].set_title("Total GPU Pipeline Time", fontweight="bold", fontsize=13)
axes[1].set_xlabel("Dataset size  N", fontsize=11)
axes[1].set_ylabel("Time (ms)", fontsize=11)
axes[1].legend(fontsize=11, facecolor="#21262D", labelcolor="#C9D1D9",
               edgecolor="#30363D")

fig.suptitle(
    f"Kernel K-Means  ·  GEMM vs SYRK Gram Matrix  ·  D={D}  K={K}  γ={GAMMA}",
    fontsize=14, color="#E6EDF3", fontweight="bold", y=1.01,
)

plt.tight_layout()
out_path = "gemm_vs_syrk_benchmark.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out_path}")


# ── 4. Summary table ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print(f"{'N':>6}  {'GEMM gram(ms)':>14}  {'SYRK gram(ms)':>14}  "
      f"{'Speedup':>8}  {'GEMM total(ms)':>15}  {'SYRK total(ms)':>15}")
print("─" * 60)
for i, n in enumerate(N_LIST):
    gg = results["gemm"]["gram_ms"][i]
    sg = results["syrk"]["gram_ms"][i]
    sp = gg / sg if sg > 0 and not np.isnan(sg) else float("nan")
    gt = results["gemm"]["total_ms"][i]
    st = results["syrk"]["total_ms"][i]
    print(f"{n:>6}  {gg:>14.2f}  {sg:>14.2f}  {sp:>8.2f}×  {gt:>15.2f}  {st:>15.2f}")
print("─" * 60)
