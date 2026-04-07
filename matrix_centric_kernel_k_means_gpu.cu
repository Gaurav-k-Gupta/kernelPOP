#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000
#define D 2
#define K_CLUSTERS 2
#define ITER 10
#define GAMMA 1.0
#define COEF 1.0
#define DEGREE 2.0

__global__ void normal_matmul_B(double* P, double* B) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && j < N) {
        double sum = 0;
        for(int d=0; d<D; d++) sum += P[i*D + d] * P[j*D + d];
        B[i*N + j] = sum;
    }
}

__global__ void compute_K(double* B, double* K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N*N) {
        K[idx] = pow(GAMMA * B[idx] + COEF, DEGREE);
    }
}

// For simplicity, matrix operations in iteration loop are done similarly
__global__ void update_dist(double* K, double* V, int* new_cluster, double* C_tilde) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        double min_dist = 1e15;
        int best_j = 0;
        for(int j=0; j<K_CLUSTERS; j++) {
            double kv_t = 0;
            for(int m=0; m<N; m++) kv_t += K[i*N + m] * V[j*N + m];

            double dist = -2.0 * kv_t + K[i*N + i] + C_tilde[j];
            if(dist < min_dist) { min_dist = dist; best_j = j; }
        }
        new_cluster[i] = best_j;
    }
}

int main() {
    double *h_P = (double*)malloc(N*D*sizeof(double));
    int *h_cluster = (int*)malloc(N*sizeof(int));
    FILE* f = fopen("data.csv", "r");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) fscanf(f, "%lf,", &h_P[i*D + j]);
        h_cluster[i] = i % K_CLUSTERS;
    }
    fclose(f);

    double *d_P, *d_B, *d_K, *d_V, *d_C_tilde;
    int *d_cluster;
    cudaMalloc(&d_P, N*D*sizeof(double));
    cudaMalloc(&d_B, N*N*sizeof(double));
    cudaMalloc(&d_K, N*N*sizeof(double));
    cudaMalloc(&d_V, K_CLUSTERS*N*sizeof(double));
    cudaMalloc(&d_C_tilde, K_CLUSTERS*sizeof(double));
    cudaMalloc(&d_cluster, N*sizeof(int));

    cudaMemcpy(d_P, h_P, N*D*sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Normal Matrix Multiplication for B = P P^T
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (N+15)/16);
    normal_matmul_B<<<blocks, threads>>>(d_P, d_B);

    int total_threads = 256;
    compute_K<<<(N*N + total_threads - 1)/total_threads, total_threads>>>(d_B, d_K);

    double *h_K = (double*)malloc(N*N*sizeof(double));
    double *h_V = (double*)malloc(K_CLUSTERS*N*sizeof(double));
    double h_C_tilde[K_CLUSTERS];
    int h_count[K_CLUSTERS];

    cudaMemcpy(h_K, d_K, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    for(int iter=0; iter<ITER; iter++) {
        for(int j=0; j<K_CLUSTERS; j++) h_count[j] = 0;
        for(int i=0; i<N; i++) h_count[h_cluster[i]]++;

        for(int j=0; j<K_CLUSTERS; j++) {
            for(int i=0; i<N; i++) {
                h_V[j*N + i] = (h_cluster[i] == j && h_count[j] > 0) ? 1.0/h_count[j] : 0.0;
            }
        }

        for(int j=0; j<K_CLUSTERS; j++) {
            double sum = 0;
            for(int u=0; u<N; u++) {
                for(int v=0; v<N; v++) sum += h_V[j*N + u] * h_K[u*N + v] * h_V[j*N + v];
            }
            h_C_tilde[j] = sum;
        }

        cudaMemcpy(d_V, h_V, K_CLUSTERS*N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_tilde, h_C_tilde, K_CLUSTERS*sizeof(double), cudaMemcpyHostToDevice);

        update_dist<<<(N+255)/256, 256>>>(d_K, d_V, d_cluster, d_C_tilde);
        cudaMemcpy(h_cluster, d_cluster, N*sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Matrix-Centric Time: %f seconds\n", ms / 1000.0);

    FILE* out = fopen("out_gpu_matrix.csv", "w");
    for(int i=0; i<N; i++) fprintf(out, "%d\n", h_cluster[i]);
    fclose(out);

    return 0;
}