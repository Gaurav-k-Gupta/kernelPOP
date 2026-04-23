#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define ITER 10

// RBF kernel:
// K(i,j) = exp(-gamma * ||x_i - x_j||^2)
// Using Gram matrix B = X X^T,
// ||x_i - x_j||^2 = B_ii - 2 B_ij + B_jj
__global__ void compute_K(double* B, double* K, int N, double GAMMA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int i = idx / N;
        int j = idx % N;

        double bii = B[i * N + i];
        double bjj = B[j * N + j];
        double bij = B[idx];

        double dist2 = bii - 2.0 * bij + bjj;
        K[idx] = exp(-GAMMA * dist2);
    }
}

// Computes E = K * V
__global__ void compute_E(double* K, int* cluster, int* count, double* E, int N, int K_CLUSTERS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < K_CLUSTERS) {
        if (count[j] == 0) {
            E[i * K_CLUSTERS + j] = 0.0;
            return;
        }
        double sum = 0.0;
        for (int m = 0; m < N; m++) {
            if (cluster[m] == j) {
                sum += K[i * N + m];
            }
        }
        E[i * K_CLUSTERS + j] = sum / count[j];
    }
}

// Computes C = diag(V^T * E)
__global__ void compute_C(double* E, int* cluster, int* count, double* C, int N, int K_CLUSTERS) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < K_CLUSTERS) {
        if (count[j] == 0) {
            C[j] = 0.0;
            return;
        }
        double sum = 0.0;
        for (int u = 0; u < N; u++) {
            if (cluster[u] == j) {
                sum += E[u * K_CLUSTERS + j];
            }
        }
        C[j] = sum / count[j];
    }
}

__global__ void update_Z(double* K, double* E, double* C, int* cluster, int N, int K_CLUSTERS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double min_dist = 1e15;
        int best_j = 0;
        double k_ii = K[i * N + i];

        for (int j = 0; j < K_CLUSTERS; j++) {
            double dist = k_ii - 2.0 * E[i * K_CLUSTERS + j] + C[j];
            if (dist < min_dist) {
                min_dist = dist;
                best_j = j;
            }
        }
        cluster[i] = best_j;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000;
    int D = (argc > 2) ? atoi(argv[2]) : 10;
    int K_CLUSTERS = (argc > 3) ? atoi(argv[3]) : 2;

    // RBF kernel parameter
    double GAMMA  = (argc > 4) ? atof(argv[4]) : 5.0;
    double COEF   = (argc > 5) ? atof(argv[5]) : 1.0;
    double DEGREE = (argc > 6) ? atof(argv[6]) : 2.0;

    double *h_P = (double*)malloc(N * D * sizeof(double));
    int *h_cluster = (int*)malloc(N * sizeof(int));

    FILE* f = fopen("data.csv", "r");
    if (!f) {
        printf("Could not open data.csv\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            if (fscanf(f, "%lf,", &h_P[i * D + j]) != 1) {
                printf("Error reading data.csv\n");
                fclose(f);
                return 1;
            }
        }
    }
    fclose(f);

    double *d_P, *d_B, *d_K, *d_E, *d_C;
    int *d_cluster, *d_count;

    cudaMalloc(&d_P, N * D * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_K, N * N * sizeof(double));
    cudaMalloc(&d_E, N * K_CLUSTERS * sizeof(double));
    cudaMalloc(&d_C, K_CLUSTERS * sizeof(double));
    cudaMalloc(&d_cluster, N * sizeof(int));
    cudaMalloc(&d_count, K_CLUSTERS * sizeof(int));

    cudaMemcpy(d_P, h_P, N * D * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Compute Gram matrix B = P * P^T
    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, D,
                &alpha,
                d_P, D,
                d_P, D,
                &beta,
                d_B, N);

    cublasDestroy(handle);

    int total_threads = 256;
    compute_K<<<(N * N + total_threads - 1) / total_threads, total_threads>>>(d_B, d_K, N, GAMMA);

    double *h_K = (double*)malloc(N * N * sizeof(double));
    cudaMemcpy(h_K, d_K, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Kernel K-Means++ initialization
    srand(42);
    int* centers = (int*)malloc(K_CLUSTERS * sizeof(int));
    centers[0] = rand() % N;

    double* min_dists = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) min_dists[i] = 1e15;

    for (int k = 1; k < K_CLUSTERS; k++) {
        int last_c = centers[k - 1];
        double sum_sq = 0.0;

        for (int i = 0; i < N; i++) {
            double dist = h_K[i * N + i] - 2.0 * h_K[i * N + last_c] + h_K[last_c * N + last_c];
            if (dist < min_dists[i]) min_dists[i] = dist;
            sum_sq += min_dists[i];
        }

        double r = ((double)rand() / RAND_MAX) * sum_sq;
        double curr_sum = 0.0;
        int next_c = N - 1;

        for (int i = 0; i < N; i++) {
            curr_sum += min_dists[i];
            if (curr_sum >= r) {
                next_c = i;
                break;
            }
        }
        centers[k] = next_c;
    }

    // Initial assignment
    for (int i = 0; i < N; i++) {
        double best_dist = 1e15;
        int best_j = 0;

        for (int k = 0; k < K_CLUSTERS; k++) {
            int c = centers[k];
            double dist = h_K[i * N + i] - 2.0 * h_K[i * N + c] + h_K[c * N + c];
            if (dist < best_dist) {
                best_dist = dist;
                best_j = k;
            }
        }
        h_cluster[i] = best_j;
    }

    free(min_dists);
    free(centers);

    int *h_count = (int*)malloc(K_CLUSTERS * sizeof(int));

    // Matrix-centric update loop
    for (int iter = 0; iter < ITER; iter++) {
        for (int j = 0; j < K_CLUSTERS; j++) h_count[j] = 0;
        for (int i = 0; i < N; i++) h_count[h_cluster[i]]++;

        cudaMemcpy(d_count, h_count, K_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster, h_cluster, N * sizeof(int), cudaMemcpyHostToDevice);

        dim3 threadsE(16, 16);
        dim3 blocksE((N + 15) / 16, (K_CLUSTERS + 15) / 16);
        compute_E<<<blocksE, threadsE>>>(d_K, d_cluster, d_count, d_E, N, K_CLUSTERS);

        int threadsC = 256;
        int blocksC = (K_CLUSTERS + 255) / 256;
        compute_C<<<blocksC, threadsC>>>(d_E, d_cluster, d_count, d_C, N, K_CLUSTERS);

        int threadsZ = 256;
        int blocksZ = (N + 255) / 256;
        update_Z<<<blocksZ, threadsZ>>>(d_K, d_E, d_C, d_cluster, N, K_CLUSTERS);

        cudaMemcpy(h_cluster, d_cluster, N * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Matrix-Centric Time: %f seconds\n", ms / 1000.0);

    FILE* out = fopen("out_gpu_matrix.csv", "w");
    for (int i = 0; i < N; i++) {
        fprintf(out, "%d\n", h_cluster[i]);
    }
    fclose(out);

    free(h_P);
    free(h_cluster);
    free(h_K);
    free(h_count);

    cudaFree(d_P);
    cudaFree(d_B);
    cudaFree(d_K);
    cudaFree(d_E);
    cudaFree(d_C);
    cudaFree(d_cluster);
    cudaFree(d_count);

    return 0;
}