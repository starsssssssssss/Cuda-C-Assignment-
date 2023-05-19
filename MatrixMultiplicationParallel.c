%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 1024

__global__ void matrixMultiplication(float* matrixA, float* matrixB, float* matrixC, int rowsA, int colsA, int colsB) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < rowsA && col < colsB) {
        float result = 0;
        for (int k = 0; k < colsA; k++) {
            result += matrixA[row * colsA + k] * matrixB[k * colsB + col];
        }
        matrixC[row * colsB + col] = result;
    }
}

int main() {
    int rowsA, colsA, colsB;
    rowsA = pow(10, 4);
    colsA = pow(10, 3);
    colsB = pow(10, 3);

    float* matrixA = (float*)malloc(sizeof(float) * rowsA * colsA);
    float* matrixB = (float*)malloc(sizeof(float) * colsA * colsB);
    float* matrixC = (float*)malloc(sizeof(float) * rowsA * colsB);

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            matrixA[i * colsA + j] = rand() % 100;
        }
    }

    for (int i = 0; i < colsA; i++) {
        for (int j = 0; j < colsB; j++) {
            matrixB[i * colsB + j] = rand() % 100;
        }
    }

    float* deviceMatrixA, * deviceMatrixB, * deviceMatrixC;
    cudaMalloc((void**)&deviceMatrixA, rowsA * colsA * sizeof(float));
    cudaMalloc((void**)&deviceMatrixB, colsA * colsB * sizeof(float));
    cudaMalloc((void**)&deviceMatrixC, rowsA * colsB * sizeof(float));

    cudaMemcpy(deviceMatrixA, matrixA, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDimensions(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimensions((rowsA - 1) / BLOCK_SIZE + 1, (colsB - 1) / BLOCK_SIZE + 1);

    clock_t startTime = clock();

    matrixMultiplication<<<gridDimensions, blockDimensions>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, rowsA, colsA, colsB);

    clock_t endTime = clock();

    cudaMemcpy(matrixC, deviceMatrixC, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    double elapsedTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

    printf("Elapsed time: %.6f seconds\n", elapsedTime);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
