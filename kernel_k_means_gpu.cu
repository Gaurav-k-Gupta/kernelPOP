#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ITER 10

__global__ void compute_kernel(double* data, double* K, int N, int D, double GAMMA, double COEF, double DEGREE) {
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

__global__ void update_clusters(double* K, int* cluster, int* count, double* C_penalty, int* new_cluster, int N, int K_CLUSTERS) {
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

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000;
    int D = (argc > 2) ? atoi(argv[2]) : 2;
    int K_CLUSTERS = (argc > 3) ? atoi(argv[3]) : 2;

    double GAMMA = 1.0, COEF = 1.0, DEGREE = 2.0;

    double *h_data = (double*)malloc(N*D*sizeof(double));
    int *h_cluster = (int*)malloc(N*sizeof(int));
    FILE* f = fopen("data.csv", "r");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) if(fscanf(f, "%lf,", &h_data[i*D + j]) != 1) {}
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    compute_kernel<<<blocks, threads>>>(d_data, d_K, N, D, GAMMA, COEF, DEGREE);

    double *h_K = (double*)malloc(N*N*sizeof(double));
    cudaMemcpy(h_K, d_K, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // Initialization Logic using computed Kernel Matrix
    srand(42);
    int* init_pts = (int*)malloc(K_CLUSTERS * sizeof(int));
    for(int j=0; j<K_CLUSTERS; j++) init_pts[j] = rand() % N;
    for(int i=0; i<N; i++) {
        double min_dist = 1e15;
        int best_j = 0;
        for(int j=0; j<K_CLUSTERS; j++) {
            int c_idx = init_pts[j];
            double dist = h_K[i*N + i] - 2.0 * h_K[i*N + c_idx] + h_K[c_idx*N + c_idx];
            if(dist < min_dist) { min_dist = dist; best_j = j; }
        }
        h_cluster[i] = best_j;
    }
    free(init_pts);

    int *h_count = (int*)malloc(K_CLUSTERS*sizeof(int));
    double *h_C_penalty = (double*)malloc(K_CLUSTERS*sizeof(double));

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
            h_C_penalty[j] = (h_count[j] > 0) ? sum / ((double)h_count[j] * h_count[j]) : 0;
        }

        cudaMemcpy(d_cluster, h_cluster, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_count, h_count, K_CLUSTERS*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_penalty, h_C_penalty, K_CLUSTERS*sizeof(double), cudaMemcpyHostToDevice);

        update_clusters<<<(N+255)/256, 256>>>(d_K, d_cluster, d_count, d_C_penalty, d_new_cluster, N, K_CLUSTERS);
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

    free(h_data); free(h_cluster); free(h_K); free(h_count); free(h_C_penalty);
    cudaFree(d_data); cudaFree(d_K); cudaFree(d_cluster); cudaFree(d_new_cluster); cudaFree(d_count); cudaFree(d_C_penalty);
    return 0;
}