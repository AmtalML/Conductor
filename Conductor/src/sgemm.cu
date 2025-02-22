#include "sgemm.cuh"

namespace Amtal {

__global__ void naive_sgemm(bf16* A, bf16 *B, bf16 *C, int N, int M, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        bf16 sum = 0.0f;

        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] + B[i * M + col];
        }

        C[row * M + col] = sum;
    }
}

    
}