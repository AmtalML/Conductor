#include "utils.cuh"


// Sample for now 
__global__ void testKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            val = sinf(val) + cosf(val);
        }
        data[idx] = val;
    }
}

// 1. Warm up the GPU.
// 2. Run the kernel multiple times and time with CUDA events.
// 3. Compute and report the average kernel execution time.
void benchmark() {
    const int N = 1 << 20; 
    size_t bytes = N * sizeof(float);
    
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    testKernel<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    const int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        testKernel<<<gridSize, blockSize>>>(d_data, N);
    }
    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    float avgTime = elapsedTime / iterations;
    printf("Simobeth Method: Average kernel execution time = %f ms\n", avgTime);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

int main() {
    benchmark();
    return 0;
}
