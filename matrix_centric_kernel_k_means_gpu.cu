#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cublas_v2.h>

#define ITER 10

// ---------------------------------------------------------------------------
// Gram Matrix via GEMM: B = P * P^T  (full N×N, no symmetry exploitation)
// Returns elapsed GPU time in milliseconds.
// ---------------------------------------------------------------------------
float compute_gram_gemm(cublasHandle_t handle,
                        double* d_P, double* d_B,
                        int N, int D)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    double alpha = 1.0, beta = 0.0;
    // B = P * P^T   →   cublasDgemm computes C = alpha * op(A) * op(B) + beta * C
    // We store P in row-major, cuBLAS expects column-major.
    // Treating P (N×D row-major) as P^T (D×N col-major):
    //   B (N×N) = P(N×D) * P^T(D×N)  →  op(A)=P^T, op(B)=P, leading dim = D
    cublasDgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,   // op(A), op(B)
                N, N, D,                     // m, n, k
                &alpha,
                d_P, D,                      // A = d_P, lda = D
                d_P, D,                      // B = d_P, ldb = D
                &beta,
                d_B, N);                     // C = d_B, ldc = N

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

// ---------------------------------------------------------------------------
// Mirror lower triangle to upper triangle kernel
// Used after SYRK which fills only the lower triangle.
// ---------------------------------------------------------------------------
__global__ void mirror_lower_to_upper(double* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N && j > i) {
        B[i * N + j] = B[j * N + i];
    }
}

// ---------------------------------------------------------------------------
// Gram Matrix via SYRK: B = P * P^T  (lower triangle only, then mirrored)
// More efficient than GEMM for symmetric results — roughly half the FLOPs.
// Returns elapsed GPU time in milliseconds.
// ---------------------------------------------------------------------------
float compute_gram_syrk(cublasHandle_t handle,
                        double* d_P, double* d_B,
                        int N, int D)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    double alpha = 1.0, beta = 0.0;
    // cublasDsyrk: C = alpha * A * A^T + beta * C  (lower fill)
    // A is d_P viewed as col-major D×N  →  op(A)=CUBLAS_OP_T gives N×N result
    cublasDsyrk(handle,
                CUBLAS_FILL_MODE_LOWER,   // fill lower triangle of C
                CUBLAS_OP_T,              // op(A): transpose so shape is N×D * D×N = N×N
                N, D,                     // n, k
                &alpha,
                d_P, D,                   // A, lda = D
                &beta,
                d_B, N);                  // C, ldc = N

    // Mirror lower → upper so the full matrix is valid for downstream kernels
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    mirror_lower_to_upper<<<blocks, threads>>>(d_B, N);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

// ---------------------------------------------------------------------------
// RBF kernel:  K(i,j) = exp(-gamma * ||x_i - x_j||^2)
// Uses identity: ||x_i - x_j||^2 = B_ii - 2*B_ij + B_jj
// ---------------------------------------------------------------------------
__global__ void compute_K(double* B, double* K, int N, double GAMMA)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int i = idx / N;
        int j = idx % N;
        double dist2 = B[i * N + i] - 2.0 * B[idx] + B[j * N + j];
        K[idx] = exp(-GAMMA * dist2);
    }
}

/* 	
𝐃=−𝟐⁢𝐊⁢𝐕𝐓+𝐏~+𝐂~
*/	
// Computes E = K * V  (V = cluster membership matrix, stored implicitly)
__global__ void compute_E(double* K, int* cluster, int* count, double* E,
                           int N, int K_CLUSTERS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < K_CLUSTERS) {
        if (count[j] == 0) { E[i * K_CLUSTERS + j] = 0.0; return; }
        double sum = 0.0;
        for (int m = 0; m < N; m++)
            if (cluster[m] == j) sum += K[i * N + m];
        E[i * K_CLUSTERS + j] = sum / count[j];
    }
}

// Computes C = diag(V^T * E)
__global__ void compute_C(double* E, int* cluster, int* count, double* C,
                           int N, int K_CLUSTERS)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < K_CLUSTERS) {
        if (count[j] == 0) { C[j] = 0.0; return; }
        double sum = 0.0;
        for (int u = 0; u < N; u++)
            if (cluster[u] == j) sum += E[u * K_CLUSTERS + j];
        C[j] = sum / count[j];
    }
}

// Assigns each point to its nearest kernel cluster centroid
__global__ void update_Z(double* K, double* E, double* C, int* cluster,
                          int N, int K_CLUSTERS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double min_dist = 1e15;
        int best_j = 0;
        double k_ii = K[i * N + i];
        for (int j = 0; j < K_CLUSTERS; j++) {
            double dist = k_ii - 2.0 * E[i * K_CLUSTERS + j] + C[j];
            if (dist < min_dist) { min_dist = dist; best_j = j; }
        }
        cluster[i] = best_j;
    }
}

// ---------------------------------------------------------------------------
// Main
// Usage: ./kernel_kmeans <N> <D> <K> <gamma> <coef> <degree> <method>
//   method: "gemm" (default) | "syrk"
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int    N         = (argc > 1) ? atoi(argv[1]) : 1000;
    int    D         = (argc > 2) ? atoi(argv[2]) : 10;
    int    K_CLUSTERS= (argc > 3) ? atoi(argv[3]) : 2;
    double GAMMA     = (argc > 4) ? atof(argv[4]) : 5.0;
    double COEF      = (argc > 5) ? atof(argv[5]) : 1.0;
    double DEGREE    = (argc > 6) ? atof(argv[6]) : 2.0;
    // 7th argument selects gram method: "gemm" or "syrk"
    int use_syrk = 0;
    if (argc > 7 && strcmp(argv[7], "syrk") == 0) use_syrk = 1;

    //8th arg -> data file
    const char* fname = (argc > 8) ? argv[8] : "data.csv";

    printf("Method: %s | N=%d  D=%d  K=%d  gamma=%.2f\n",
           use_syrk ? "SYRK" : "GEMM", N, D, K_CLUSTERS, GAMMA);

    // ---- Host allocations & data load ----
    double* h_P       = (double*)malloc(N * D * sizeof(double));
    int*    h_cluster = (int*)   malloc(N      * sizeof(int));

    FILE* f = fopen(fname, "r");
    if (!f) { printf("Could not open <%s>\n", fname); return 1; }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (fscanf(f, "%lf,", &h_P[i * D + j]) != 1) {
                printf("Error reading data.csv\n"); fclose(f); return 1;
            }
    fclose(f);

    // ---- Device allocations ----
    double *d_P, *d_B, *d_K, *d_E, *d_C;
    int    *d_cluster, *d_count;

    cudaMalloc(&d_P,       N * D         * sizeof(double));
    cudaMalloc(&d_B,       N * N         * sizeof(double));
    cudaMalloc(&d_K,       N * N         * sizeof(double));
    cudaMalloc(&d_E,       N * K_CLUSTERS* sizeof(double));
    cudaMalloc(&d_C,       K_CLUSTERS    * sizeof(double));
    cudaMalloc(&d_cluster, N             * sizeof(int));
    cudaMalloc(&d_count,   K_CLUSTERS    * sizeof(int));

    cudaMemcpy(d_P, h_P, N * D * sizeof(double), cudaMemcpyHostToDevice);

    // ---- Overall timing ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ---- Gram matrix (GEMM or SYRK) ----
    cublasHandle_t handle;
    cublasCreate(&handle);

    float gram_ms = 0.0f;
    if (use_syrk)
        gram_ms = compute_gram_syrk(handle, d_P, d_B, N, D);
    else
        gram_ms = compute_gram_gemm(handle, d_P, d_B, N, D);

    printf("Gram matrix (%s) time: %.4f ms\n",
           use_syrk ? "SYRK" : "GEMM", gram_ms);

    cublasDestroy(handle);

    // ---- RBF kernel ----
    int total_threads = 256;
    compute_K<<<(N * N + total_threads - 1) / total_threads, total_threads>>>(
        d_B, d_K, N, GAMMA);

    // ---- Copy K to host for K-Means++ init ----
    double* h_K = (double*)malloc(N * N * sizeof(double));
    cudaMemcpy(h_K, d_K, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // ---- Kernel K-Means++ initialisation ----
    srand(42);
    int*    centers   = (int*)   malloc(K_CLUSTERS * sizeof(int));
    double* min_dists = (double*)malloc(N          * sizeof(double));

    centers[0] = rand() % N;
    for (int i = 0; i < N; i++) min_dists[i] = 1e15;

    for (int k = 1; k < K_CLUSTERS; k++) {
        int last_c  = centers[k - 1];
        double sum_sq = 0.0;
        for (int i = 0; i < N; i++) {
            double dist = h_K[i*N+i] - 2.0*h_K[i*N+last_c] + h_K[last_c*N+last_c];
            if (dist < min_dists[i]) min_dists[i] = dist;
            sum_sq += min_dists[i];
        }
        double r = ((double)rand() / RAND_MAX) * sum_sq;
        double curr_sum = 0.0;
        int next_c = N - 1;
        for (int i = 0; i < N; i++) {
            curr_sum += min_dists[i];
            if (curr_sum >= r) { next_c = i; break; }
        }
        centers[k] = next_c;
    }

    // Initial cluster assignment
    for (int i = 0; i < N; i++) {
        double best_dist = 1e15;
        int best_j = 0;
        for (int k = 0; k < K_CLUSTERS; k++) {
            int c = centers[k];
            double dist = h_K[i*N+i] - 2.0*h_K[i*N+c] + h_K[c*N+c];
            if (dist < best_dist) { best_dist = dist; best_j = k; }
        }
        h_cluster[i] = best_j;
    }

    free(min_dists);
    free(centers);

    int* h_count = (int*)malloc(K_CLUSTERS * sizeof(int));

    // ---- K-Means update loop ----
    for (int iter = 0; iter < ITER; iter++) {
        for (int j = 0; j < K_CLUSTERS; j++) h_count[j] = 0;
        for (int i = 0; i < N; i++) h_count[h_cluster[i]]++;

        cudaMemcpy(d_count,   h_count,   K_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster, h_cluster, N          * sizeof(int), cudaMemcpyHostToDevice);

        dim3 threadsE(16, 16);
        dim3 blocksE((N + 15) / 16, (K_CLUSTERS + 15) / 16);
        compute_E<<<blocksE, threadsE>>>(d_K, d_cluster, d_count, d_E, N, K_CLUSTERS);

        compute_C<<<(K_CLUSTERS + 255) / 256, 256>>>(
            d_E, d_cluster, d_count, d_C, N, K_CLUSTERS);

        update_Z<<<(N + 255) / 256, 256>>>(
            d_K, d_E, d_C, d_cluster, N, K_CLUSTERS);

        cudaMemcpy(h_cluster, d_cluster, N * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    printf("Total GPU time (%s): %.4f seconds\n",
           use_syrk ? "SYRK" : "GEMM", total_ms / 1000.0f);

    printf("%.4f %.4f\n", gram_ms , total_ms);

    // ---- Write timing line for Python script ----
    // Format:  method,gram_ms,total_ms
    FILE* tm = fopen("timing.csv", "a");
    if (tm) {
        fprintf(tm, "%s,%.4f,%.4f\n",
                use_syrk ? "SYRK" : "GEMM", gram_ms, total_ms);
        fclose(tm);
    }

    // ---- Write cluster assignments ----
    const char* out_name = use_syrk ? "out_gpu_syrk.csv" : "out_gpu_gemm.csv";
    FILE* out = fopen(out_name, "w");
    for (int i = 0; i < N; i++) fprintf(out, "%d\n", h_cluster[i]);
    fclose(out);

    // ---- Cleanup ----
    free(h_P); free(h_cluster); free(h_K); free(h_count);
    cudaFree(d_P); cudaFree(d_B); cudaFree(d_K);
    cudaFree(d_E); cudaFree(d_C);
    cudaFree(d_cluster); cudaFree(d_count);

    return 0;
}
