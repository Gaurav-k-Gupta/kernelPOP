/*
 * OPTIMIZED Matrix-Centric Kernel K-Means with cuSPARSE SpMM
 * ============================================================
 * Key optimization: V^T is N×k with exactly 1 non-zero per row.
 *   row_ptr = [0,1,2,...,N]  (constant, precomputed once)
 *   col_ind = cluster[]      (already exists!)
 *   values  = 1/count[c[i]]  (one kernel per iteration)
 *
 * E = K * V^T  →  E^T = V * K  →  cusparseSpMM(TRANSPOSE, V^T, K)
 *
 * Compile:
 *   nvcc matrix_centric_kernel_k_means_gpu_optimized.cu -o gpu_matrix_opt \
 *        -O3 -Wno-deprecated-gpu-targets -lcublas -lcusparse
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cublas_v2.h>
#include <cusparse.h>

#define ITER 10

// ==================== GRAM MATRIX (unchanged) ====================

float compute_gram_gemm(cublasHandle_t handle,
                        double* d_P, double* d_B, int N, int D)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, D, &alpha, d_P, D, d_P, D, &beta, d_B, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

__global__ void mirror_lower_to_upper(double* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N && j > i) B[i * N + j] = B[j * N + i];
}

float compute_gram_syrk(cublasHandle_t handle,
                        double* d_P, double* d_B, int N, int D)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    double alpha = 1.0, beta = 0.0;
    cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                N, D, &alpha, d_P, D, &beta, d_B, N);
    dim3 thr(16,16), blk((N+15)/16,(N+15)/16);
    mirror_lower_to_upper<<<blk,thr>>>(d_B, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

// ==================== HELPER KERNELS ====================

// Extract diagonal: out[i] = M[i*N+i]
__global__ void extract_diagonal(double* M, double* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = M[i * N + i];
}

// RBF kernel using precomputed diagonal
__global__ void compute_K_opt(double* B, double* Bdiag, double* K,
                              int N, double GAMMA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int i = idx / N, j = idx % N;
        double dist2 = Bdiag[i] - 2.0 * B[idx] + Bdiag[j];
        K[idx] = exp(-GAMMA * dist2);
    }
}

// Cluster counts via atomicAdd
__global__ void compute_counts_gpu(int* cluster, int* count, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(&count[cluster[i]], 1);
}

// Build V^T values: vt_val[i] = 1.0 / count[cluster[i]]
// (row_ptr is constant [0,1,...,N], col_ind IS d_cluster)
__global__ void build_vt_values(int* cluster, int* count,
                                double* vt_val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = cluster[i];
        vt_val[i] = (count[c] > 0) ? 1.0 / count[c] : 0.0;
    }
}

// Initialize row_ptr = [0, 1, 2, ..., N] (called once)
__global__ void init_row_ptr(int* row_ptr, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= N) row_ptr[i] = i;
}

// C[j] = sum_{i in C_j} E[i,j] / count[j]  (k is small, simple kernel)
__global__ void compute_C_sparse(int* cluster, int* count, double* E,
                                 double* C, int N, int K_CLUSTERS) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < K_CLUSTERS) {
        if (count[j] == 0) { C[j] = 0.0; return; }
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            if (cluster[i] == j) sum += E[i * K_CLUSTERS + j];
        C[j] = sum / count[j];
    }
}

// Argmin assignment using precomputed Kdiag
__global__ void update_Z(double* Kdiag, double* E, double* C,
                         int* cluster, int N, int K_CLUSTERS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double min_dist = 1e15; int best_j = 0;
        double k_ii = Kdiag[i];
        for (int j = 0; j < K_CLUSTERS; j++) {
            double dist = k_ii - 2.0 * E[i * K_CLUSTERS + j] + C[j];
            if (dist < min_dist) { min_dist = dist; best_j = j; }
        }
        cluster[i] = best_j;
    }
}

// ==================== MAIN ====================

int main(int argc, char** argv)
{
    int    N          = (argc > 1) ? atoi(argv[1]) : 1000;
    int    D          = (argc > 2) ? atoi(argv[2]) : 10;
    int    K_CLUSTERS = (argc > 3) ? atoi(argv[3]) : 2;
    double GAMMA      = (argc > 4) ? atof(argv[4]) : 5.0;
    double COEF       = (argc > 5) ? atof(argv[5]) : 1.0;
    double DEGREE     = (argc > 6) ? atof(argv[6]) : 2.0;
    int use_syrk = (argc > 7 && strcmp(argv[7], "syrk") == 0);
    const char* fname = (argc > 8) ? argv[8] : "data.csv";

    printf("Method: %s (SpMM-OPT) | N=%d D=%d K=%d gamma=%.2f\n",
           use_syrk ? "SYRK" : "GEMM", N, D, K_CLUSTERS, GAMMA);

    // ---- Load data ----
    double* h_P = (double*)malloc(N * D * sizeof(double));
    int*  h_cluster = (int*)malloc(N * sizeof(int));
    FILE* f = fopen(fname, "r");
    if (!f) { printf("Cannot open <%s>\n", fname); return 1; }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (fscanf(f, "%lf,", &h_P[i*D+j]) != 1) {
                printf("Read error\n"); fclose(f); return 1;
            }
    fclose(f);

    // ---- Device allocations ----
    double *d_P, *d_B, *d_K, *d_E, *d_C;
    double *d_Bdiag, *d_Kdiag;
    double *d_vt_values;          // SpMM: V^T values
    int    *d_vt_row_ptr;         // SpMM: V^T row_ptr [0,1,...,N]
    int    *d_cluster, *d_count;

    cudaMalloc(&d_P,           N * D          * sizeof(double));
    cudaMalloc(&d_B,           N * N          * sizeof(double));
    cudaMalloc(&d_K,           N * N          * sizeof(double));
    cudaMalloc(&d_E,           N * K_CLUSTERS * sizeof(double));
    cudaMalloc(&d_C,           K_CLUSTERS     * sizeof(double));
    cudaMalloc(&d_Bdiag,       N              * sizeof(double));
    cudaMalloc(&d_Kdiag,       N              * sizeof(double));
    cudaMalloc(&d_vt_values,   N              * sizeof(double));
    cudaMalloc(&d_vt_row_ptr,  (N + 1)        * sizeof(int));
    cudaMalloc(&d_cluster,     N              * sizeof(int));
    cudaMalloc(&d_count,       K_CLUSTERS     * sizeof(int));

    cudaMemcpy(d_P, h_P, N * D * sizeof(double), cudaMemcpyHostToDevice);

    int tpb = 256;

    // ---- Initialize V^T row_ptr = [0,1,2,...,N] ONCE ----
    init_row_ptr<<<(N + 2 + tpb - 1) / tpb, tpb>>>(d_vt_row_ptr, N);

    // ---- Timing ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ---- Gram matrix ----
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float gram_ms = use_syrk
        ? compute_gram_syrk(cublas_handle, d_P, d_B, N, D)
        : compute_gram_gemm(cublas_handle, d_P, d_B, N, D);
    printf("Gram matrix (%s) time: %.4f ms\n", use_syrk?"SYRK":"GEMM", gram_ms);

    // ---- Diagonal + RBF ----
    extract_diagonal<<<(N+tpb-1)/tpb, tpb>>>(d_B, d_Bdiag, N);
    compute_K_opt<<<(N*N+tpb-1)/tpb, tpb>>>(d_B, d_Bdiag, d_K, N, GAMMA);
    extract_diagonal<<<(N+tpb-1)/tpb, tpb>>>(d_K, d_Kdiag, N);

    // ---- K-Means++ init (on host, runs once) ----
    double* h_K = (double*)malloc(N * N * sizeof(double));
    cudaMemcpy(h_K, d_K, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    srand(42);
    int* centers = (int*)malloc(K_CLUSTERS * sizeof(int));
    double* min_dists = (double*)malloc(N * sizeof(double));
    centers[0] = rand() % N;
    for (int i = 0; i < N; i++) min_dists[i] = 1e15;
    for (int k = 1; k < K_CLUSTERS; k++) {
        int lc = centers[k-1]; double ss = 0;
        for (int i = 0; i < N; i++) {
            double d = h_K[i*N+i] - 2*h_K[i*N+lc] + h_K[lc*N+lc];
            if (d < min_dists[i]) min_dists[i] = d;
            ss += min_dists[i];
        }
        double r = ((double)rand()/RAND_MAX)*ss, cs = 0;
        int nc = N-1;
        for (int i = 0; i < N; i++) { cs += min_dists[i]; if (cs >= r) { nc=i; break; } }
        centers[k] = nc;
    }
    for (int i = 0; i < N; i++) {
        double bd = 1e15; int bj = 0;
        for (int k = 0; k < K_CLUSTERS; k++) {
            int c = centers[k];
            double d = h_K[i*N+i] - 2*h_K[i*N+c] + h_K[c*N+c];
            if (d < bd) { bd = d; bj = k; }
        }
        h_cluster[i] = bj;
    }
    free(min_dists); free(centers); free(h_K);
    cudaMemcpy(d_cluster, h_cluster, N*sizeof(int), cudaMemcpyHostToDevice);

    // ---- Setup cuSPARSE ----
    cusparseHandle_t sp_handle;
    cusparseCreate(&sp_handle);

    // V^T CSR descriptor: N×K_CLUSTERS, nnz=N
    // row_ptr = d_vt_row_ptr [0,1,...,N] (constant)
    // col_ind = d_cluster (updated each iter automatically)
    // values  = d_vt_values (rebuilt each iter)
    cusparseSpMatDescr_t matVT;
    cusparseCreateCsr(&matVT, N, K_CLUSTERS, N,
                      d_vt_row_ptr, d_cluster, d_vt_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // K dense: N×N, row-major
    cusparseDnMatDescr_t matK;
    cusparseCreateDnMat(&matK, N, N, N, d_K, CUDA_R_64F, CUSPARSE_ORDER_ROW);

    // E output: described as E^T (K_CLUSTERS × N) column-major ld=K_CLUSTERS
    // Column-major E^T[j + i*K] = row-major E[i*K + j]  ✓
    cusparseDnMatDescr_t matE;
    cusparseCreateDnMat(&matE, K_CLUSTERS, N, K_CLUSTERS, d_E,
                        CUDA_R_64F, CUSPARSE_ORDER_COL);

    // Allocate SpMM workspace buffer
    double alpha_sp = 1.0, beta_sp = 0.0;
    size_t bufferSize = 0;

    // Dry run to get buffer size
    // E^T = op(V^T) * K = V * K   (TRANSPOSE on V^T gives V)
    cusparseSpMM_bufferSize(sp_handle,
                            CUSPARSE_OPERATION_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_sp, matVT, matK, &beta_sp, matE,
                            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
                            &bufferSize);
    void* d_spmm_buffer;
    cudaMalloc(&d_spmm_buffer, bufferSize);
    printf("SpMM buffer: %zu bytes\n", bufferSize);

    // ---- K-Means iteration loop ----
    for (int iter = 0; iter < ITER; iter++) {

        // 1. Cluster counts on GPU
        cudaMemset(d_count, 0, K_CLUSTERS * sizeof(int));
        compute_counts_gpu<<<(N+tpb-1)/tpb, tpb>>>(d_cluster, d_count, N);

        // 2. Build V^T values (row_ptr constant, col_ind = d_cluster)
        build_vt_values<<<(N+tpb-1)/tpb, tpb>>>(d_cluster, d_count,
                                                  d_vt_values, N);

        // 3. E = K * V^T via cuSPARSE SpMM
        //    Computed as: E^T = V * K using TRANSPOSE on V^T CSR
        //    Output in col-major = row-major E
        cusparseSpMM(sp_handle,
                     CUSPARSE_OPERATION_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_sp, matVT, matK, &beta_sp, matE,
                     CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
                     d_spmm_buffer);

        // 4. C[j] = (1/count[j]) * sum_{i in C_j} E[i,j]
        compute_C_sparse<<<(K_CLUSTERS+tpb-1)/tpb, tpb>>>(
            d_cluster, d_count, d_E, d_C, N, K_CLUSTERS);

        // 5. Argmin assignment
        update_Z<<<(N+tpb-1)/tpb, tpb>>>(d_Kdiag, d_E, d_C, d_cluster,
                                           N, K_CLUSTERS);
    }

    cudaMemcpy(h_cluster, d_cluster, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float total_ms; cudaEventElapsedTime(&total_ms, start, stop);
    printf("Total GPU time (%s): %.4f seconds\n",
           use_syrk?"SYRK":"GEMM", total_ms/1000.0f);
    printf("%.4f %.4f\n", gram_ms, total_ms);

    // ---- Output ----
    FILE* tm = fopen("timing.csv", "a");
    if (tm) { fprintf(tm, "%s,%.4f,%.4f\n", use_syrk?"SYRK":"GEMM", gram_ms, total_ms); fclose(tm); }

    const char* out_name = use_syrk ? "out_gpu_syrk.csv" : "out_gpu_gemm.csv";
    FILE* out = fopen(out_name, "w");
    for (int i = 0; i < N; i++) fprintf(out, "%d\n", h_cluster[i]);
    fclose(out);

    // ---- Cleanup ----
    cusparseDestroySpMat(matVT);
    cusparseDestroyDnMat(matK);
    cusparseDestroyDnMat(matE);
    cusparseDestroy(sp_handle);
    cublasDestroy(cublas_handle);
    cudaFree(d_spmm_buffer);
    free(h_P); free(h_cluster);
    cudaFree(d_P); cudaFree(d_B); cudaFree(d_K);
    cudaFree(d_E); cudaFree(d_C);
    cudaFree(d_Bdiag); cudaFree(d_Kdiag);
    cudaFree(d_vt_values); cudaFree(d_vt_row_ptr);
    cudaFree(d_cluster); cudaFree(d_count);
    return 0;
}
