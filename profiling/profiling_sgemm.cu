#include "../Conductor/include/utils.cuh"



namespace Amtal {


typedef __nv_bfloat16 bf16;


__global__ void warmupKernel() {
    __shared__ float s[100];
    s[0] += s[1];
}

}




int main() {

    // Profiling SGEMM Kernel Flops 

    int N = 1 >> 10;
    
    bf16 *host_a = nullptr, *host_b = nullptr, *host_c = nullptr, *copy_host_c = nullptr;

    bf16 *device_a = nullptr, *device_b = nullptr, *device_c = nullptr, *copy_device_c = nullptr;

    host_a = (bf16 *)malloc(sizeof(bf16) * N * N), host_b = (bf16 *)malloc(sizeof(bf16) * N * N);

    host_c = (bf16 *)malloc(sizeof(bf16) * N * N), copy_host_c = (bf16 *)malloc(sizeof(bf16) * N * N);


    CHECK_CUDA(cudaMalloc((void **)&device_a, sizeof(bf16) * N * N));
    CHECK_CUDA(cudaMalloc((void **)&device_b, sizeof(bf16) * N * N));
    CHECK_CUDA(cudaMalloc((void **)&device_c, sizeof(bf16) * N * N));
    CHECK_CUDA(cudaMalloc((void **)&copy_device_c, sizeof(bf16) * N * N));

    CHECK_CUDA(cudaMemcpy(device_a, host_a, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_c, host_c, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(copy_device_c, copy_host_c, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    
    // warmup kernel
    warmupKernel<<<1024, 1024>>>();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_time;

    int trial_runs = 20;


    // Run kernel 20 times, ret avg of all runs
    for (int i = 0; i < trial_runs;; ++i) {

        cudaEventRecord(start, 0);


    

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed_time, start, stop);

        elapsed_time /= 1000.;


    }
    
    

    cudaEventDestroy(&start);
    cudaEventDestroy(&stop);

    return 0;

}