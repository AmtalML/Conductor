// engine/include/kernels.cuh
#pragma once
__global__ void placeholder_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx];
    }
}
