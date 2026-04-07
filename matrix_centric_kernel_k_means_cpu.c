#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 1000
#define D 2
#define K_CLUSTERS 2
#define ITER 10
#define GAMMA 1.0
#define COEF 1.0
#define DEGREE 2.0

int main() {
    double* P = (double*)malloc(N * D * sizeof(double));
    double* B = (double*)malloc(N * N * sizeof(double));
    double* K = (double*)malloc(N * N * sizeof(double));
    double* V = (double*)malloc(K_CLUSTERS * N * sizeof(double));
    double* KV_T = (double*)malloc(N * K_CLUSTERS * sizeof(double));
    double* C_tilde = (double*)malloc(K_CLUSTERS * sizeof(double));
    int* cluster = (int*)malloc(N * sizeof(int));
    int* count = (int*)malloc(K_CLUSTERS * sizeof(int));

    FILE* f = fopen("data.csv", "r");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) fscanf(f, "%lf,", &P[i*D + j]);
        cluster[i] = i % K_CLUSTERS;
    }
    fclose(f);

    clock_t start = clock();

    // Find B using normal matrix multiplication
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            double sum = 0;
            for(int d=0; d<D; d++) sum += P[i*D + d] * P[j*D + d];
            B[i*N + j] = sum;
        }
    }

    // Compute K
    for(int i=0; i<N*N; i++) {
        K[i] = pow(GAMMA * B[i] + COEF, DEGREE);
    }

    for(int iter=0; iter<ITER; iter++) {
        for(int j=0; j<K_CLUSTERS; j++) count[j] = 0;
        for(int i=0; i<N; i++) count[cluster[i]]++;

        // Form V
        for(int j=0; j<K_CLUSTERS; j++) {
            for(int i=0; i<N; i++) {
                V[j*N + i] = (cluster[i] == j && count[j] > 0) ? 1.0 / count[j] : 0.0;
            }
        }

        // KV_T = K * V^T using normal matrix multiplication
        for(int i=0; i<N; i++) {
            for(int j=0; j<K_CLUSTERS; j++) {
                double sum = 0;
                for(int m=0; m<N; m++) sum += K[i*N + m] * V[j*N + m];
                KV_T[i*K_CLUSTERS + j] = sum;
            }
        }

        // C_tilde
        for(int j=0; j<K_CLUSTERS; j++) {
            double sum = 0;
            for(int u=0; u<N; u++) {
                for(int v=0; v<N; v++) {
                    sum += V[j*N + u] * K[u*N + v] * V[j*N + v];
                }
            }
            C_tilde[j] = sum;
        }

        // D = -2 KV_T + P_tilde + C_tilde
        for(int i=0; i<N; i++) {
            int best_j = -1;
            double min_dist = 1e15;
            for(int j=0; j<K_CLUSTERS; j++) {
                double dist = -2.0 * KV_T[i*K_CLUSTERS + j] + K[i*N + i] + C_tilde[j];
                if(dist < min_dist) { min_dist = dist; best_j = j; }
            }
            cluster[i] = best_j;
        }
    }

    clock_t end = clock();
    printf("CPU Matrix-Centric Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen("out_cpu_matrix.csv", "w");
    for(int i=0; i<N; i++) fprintf(out, "%d\n", cluster[i]);
    fclose(out);

    free(P); free(B); free(K); free(V); free(KV_T); free(C_tilde); free(cluster); free(count);
    return 0;
}