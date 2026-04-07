#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ITER 10

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000;
    int D = (argc > 2) ? atoi(argv[2]) : 2;
    int K_CLUSTERS = (argc > 3) ? atoi(argv[3]) : 2;

    double GAMMA = 1.0, COEF = 1.0, DEGREE = 2.0;

    double* data = (double*)malloc(N * D * sizeof(double));
    double* K = (double*)malloc(N * N * sizeof(double));
    int* cluster = (int*)malloc(N * sizeof(int));
    int* count = (int*)malloc(K_CLUSTERS * sizeof(int));
    double* C_penalty = (double*)malloc(K_CLUSTERS * sizeof(double));

    FILE* f = fopen("data.csv", "r");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            if(fscanf(f, "%lf,", &data[i*D + j]) != 1) { /* handle warning */ }
        }
    }
    fclose(f);

    clock_t start = clock();

    // Compute Kernel Matrix (Polynomial Kernel)
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            double dot = 0;
            for(int f=0; f<D; f++) dot += data[i*D + f] * data[j*D + f];
            K[i*N + j] = pow(GAMMA * dot + COEF, DEGREE);
        }
    }

    // Better Initialization (Random Points breaks symmetric failures)
    srand(42);
    int* init_pts = (int*)malloc(K_CLUSTERS * sizeof(int));
    for(int j=0; j<K_CLUSTERS; j++) init_pts[j] = rand() % N;

    for(int i=0; i<N; i++) {
        double min_dist = 1e15;
        int best_j = 0;
        for(int j=0; j<K_CLUSTERS; j++) {
            int c_idx = init_pts[j];
            double dist = K[i*N + i] - 2.0 * K[i*N + c_idx] + K[c_idx*N + c_idx];
            if(dist < min_dist) { min_dist = dist; best_j = j; }
        }
        cluster[i] = best_j;
    }
    free(init_pts);

    for(int iter=0; iter<ITER; iter++) {
        for(int j=0; j<K_CLUSTERS; j++) count[j] = 0;
        for(int i=0; i<N; i++) count[cluster[i]]++;

        // Calculate cluster penalties
        for(int j=0; j<K_CLUSTERS; j++) {
            double sum = 0;
            for(int m=0; m<N; m++) {
                if(cluster[m] == j) {
                    for(int l=0; l<N; l++) if(cluster[l] == j) sum += K[m*N + l];
                }
            }
            C_penalty[j] = (count[j] > 0) ? sum / ((double)count[j] * count[j]) : 0;
        }

        // Assign points
        for(int i=0; i<N; i++) {
            double min_dist = 1e15;
            int best_j = 0;
            for(int j=0; j<K_CLUSTERS; j++) {
                if(count[j] == 0) continue;
                double term2 = 0;
                for(int m=0; m<N; m++) if(cluster[m] == j) term2 += K[i*N + m];
                term2 = -2.0 * term2 / count[j];
                double dist = K[i*N + i] + term2 + C_penalty[j];
                if(dist < min_dist) {
                    min_dist = dist;
                    best_j = j;
                }
            }
            cluster[i] = best_j;
        }
    }

    clock_t end = clock();
    printf("CPU Normal Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen("out_cpu_normal.csv", "w");
    for(int i=0; i<N; i++) fprintf(out, "%d\n", cluster[i]);
    fclose(out);

    free(data); free(K); free(cluster); free(count); free(C_penalty);
    return 0;
}