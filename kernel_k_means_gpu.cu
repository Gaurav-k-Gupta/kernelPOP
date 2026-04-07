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

__global__ void compute_kernel(double* data, double* K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        double dot = 0;
        for (int f = 0; f < D; f++) {
            dot += data[i * D + f] * data[j * D + f];
        }
        K[i * N + j] = pow(GAMMA * dot + COEF, DEGREE);
    }
}

// Simplified GPU kernel mapping iterations for Normal Kernel K-means
__global__ void update_clusters(double* K, int* cluster, int* count, double* C_penalty, int* new_cluster) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double min_dist = 1e15;
        int best_j = 0;
        for(int j=0; j<K_CLUSTERS; j++) {
            if(count[j] == 0) continue;
            double term2 = 0;
            for(int m=0; m<N; m++) {
                if(cluster[m] == j) term2 += K[i*N + m];
            }
            term2 = -2.0 * term2 / count[j];
            double dist = K[i*N + i] + term2 + C_penalty[j];
            if(dist < min_dist) {
                min_dist = dist;
                best_j = j;
            }
        }
        new_cluster[i] = best_j;
    }
}

int main() {
    double *h_data = (double*)malloc(N*D*sizeof(double));
    int *h_cluster = (int*)malloc(N*sizeof(int));
    FILE* f = fopen("data.csv", "r");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) fscanf(f, "%lf,", &h_data[i*D + j]);
        h_cluster[i] = i % K_CLUSTERS;
    }
    fclose(f);

    double *d_data, *d_K, *d_C_penalty;
    int *d_cluster, *d_count, *d_new_cluster;
    cudaMalloc(&d_data, N*D*sizeof(double));
    cudaMalloc(&d_K, N*N*sizeof(double));
    cudaMalloc(&d_cluster, N*sizeof(int));
    cudaMalloc(&d_new_cluster, N*sizeof(int));
    cudaMalloc(&d_count, K_CLUSTERS*sizeof(int));
    cudaMalloc(&d_C_penalty, K_CLUSTERS*sizeof(double));

    cudaMemcpy(d_data, h_data, N*D*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster, h_cluster, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    compute_kernel<<<blocks, threads>>>(d_data, d_K);

    int h_count[K_CLUSTERS];
    double h_C_penalty[K_CLUSTERS];
    double *h_K = (double*)malloc(N*N*sizeof(double));
    cudaMemcpy(h_K, d_K, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    for (int iter = 0; iter < ITER; iter++) {
        for(int j=0; j<K_CLUSTERS; j++) h_count[j] = 0;
        for(int i=0; i<N; i++) h_count[h_cluster[i]]++;

        for(int j=0; j<K_CLUSTERS; j++) {
            double sum = 0;
            for(int m=0; m<N; m++) {
                if(h_cluster[m] == j) {
                    for(int l=0; l<N; l++) if(h_cluster[l] == j) sum += h_K[m*N + l];
                }
            }
            h_C_penalty[j] = (h_count[j] > 0) ? sum / (h_count[j] * h_count[j]) : 0;
        }

        cudaMemcpy(d_cluster, h_cluster, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_count, h_count, K_CLUSTERS*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_penalty, h_C_penalty, K_CLUSTERS*sizeof(double), cudaMemcpyHostToDevice);

        update_clusters<<<(N+255)/256, 256>>>(d_K, d_cluster, d_count, d_C_penalty, d_new_cluster);
        cudaMemcpy(h_cluster, d_new_cluster, N*sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Normal Time: %f seconds\n", ms / 1000.0);

    FILE* out = fopen("out_gpu_normal.csv", "w");
    for(int i=0; i<N; i++) fprintf(out, "%d\n", h_cluster[i]);
    fclose(out);

    return 0;
}