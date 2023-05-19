#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 4

__global__ void matrixMultiplication(float *A, float *B, float *C, int r1, int c1, int c2) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    float result = 0.0;
    for (int t = 0; t < (c1 - 1) / TILE_SIZE + 1; ++t) {
        if (row < r1 && t * TILE_SIZE + threadIdx.y < c1) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * c1 + t * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < c2 && t * TILE_SIZE + threadIdx.y < c1) {
            shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * c2 + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            result += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < r1 && col < c2) {
        C[row * c2 + col] = result;
    }
}

int main() {
    int r1, c1, c2;
    r1 = pow(10, 4);
    c1 = pow(10, 3);
    c2 = pow(10, 3);

    float *A = (float*)malloc(sizeof(float) * r1 * c1);
    float *B = (float*)malloc(sizeof(float) * c1 * c2);
    float *C = (float*)malloc(sizeof(float) * r1 * c2);
   
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, r1 * c1 * sizeof(float));
    cudaMalloc((void**)&d_B, c1 * c2 * sizeof(float));
    cudaMalloc((void**)&d_C, r1 * c2 * sizeof(float));
    cudaMemcpy(d_A, A, r1 * c1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, c1 * c2 * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((r1 - 1) / BLOCK_SIZE + 1, (c2 - 1) / BLOCK_SIZE + 1);

    
    clock_t start = clock();
    matrixMultiplication<<<grid_dim, block_dim>>>(d_A, d_B, d_C, r1, c1, c2);
    clock_t end = clock();

    
        cudaMemcpy(C, d_C, r1 * c2 * sizeof(float), cudaMemcpyDeviceToHost);

  
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Elapsed time: %.6f seconds\n", elapsed_time);

   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

   
    free(A);
    free(B);
    free(C);

    return 0;
}

