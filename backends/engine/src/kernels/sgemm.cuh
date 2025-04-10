#include "../include/utils.cuh"



__global__ void sgemm_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    uint temp = 0;
    if (x < M && y < N) {
       
        for (uint i = 0; i < K; ++i) {
            temp += A[x * K + i] * B[N * i + y];
        }
    }

    C[x * N + y] = temp * alpha + beta * C[x * N + y];
}


