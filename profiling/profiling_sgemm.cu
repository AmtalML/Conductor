#include "../Conductor/include/utils.cuh"


std::default_random_engine generator(42);
cublasHandle_t cublas_handle;

namespace Amtal {

void run_cublas_sgemm_bf16(bf16 *A, bf16 *B, bf16 *C, int M, int N, int K, float alpha, float beta) {
    // For A: stored row-major (M×K) so that
    // A^T (interpreted as col-major) becomes M×K.
    // Use lda = K and for C (M×N col-major) use ldc = M.
    cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                         M, N, K, &alpha,
                                         A, CUDA_R_16BF, K,
                                         B, CUDA_R_16BF, K, &beta,
                                         C, CUDA_R_16BF, M,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << "\n";
        exit(1);
    }
}

// warming up hardware and avoiding undefined behavior
__global__ void warmupKernel() {
    __shared__ float s[100];
    s[0] = 0.0f; s[1] = 0.0f;
    s[0] += s[1];
}

void randomize_matrix(bf16 *matrix, int total_elements) {
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < total_elements; ++i) {
        matrix[i] = __float2bfloat16(distribution(generator));
    }
}

} 

int main() {

    if(cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS){
        std::cout << "Failed to create cuBLAS handle.\n";
        exit(1);
    }

    int N = 1 << 10;
    int M = N, K = N;
    float alpha = 1.0f, beta = 0.0f;

    bf16 *host_a = (bf16 *)malloc(sizeof(bf16) * N * N);
    bf16 *host_b = (bf16 *)malloc(sizeof(bf16) * N * N);
    bf16 *host_c = (bf16 *)malloc(sizeof(bf16) * N * N);
    bf16 *copy_host_c = (bf16 *)malloc(sizeof(bf16) * N * N);

    Amtal::randomize_matrix(host_a, N * N);
    Amtal::randomize_matrix(host_b, N * N);
    memset(host_c, 0, sizeof(bf16) * N * N);
    memset(copy_host_c, 0, sizeof(bf16) * N * N);

    bf16 *device_a = nullptr, *device_b = nullptr, *device_c = nullptr, *copy_device_c = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&device_a, sizeof(bf16) * N * N));
    CHECK_CUDA(cudaMalloc((void **)&device_b, sizeof(bf16) * N * N));
    CHECK_CUDA(cudaMalloc((void **)&device_c, sizeof(bf16) * N * N));
    CHECK_CUDA(cudaMalloc((void **)&copy_device_c, sizeof(bf16) * N * N));

    CHECK_CUDA(cudaMemcpy(device_a, host_a, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_c, host_c, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(copy_device_c, copy_host_c, sizeof(bf16) * N * N, cudaMemcpyHostToDevice));

    Amtal::warmupKernel<<<1024, 1024>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time = 0.0f;
    int trial_runs = 20;

    cudaDeviceSynchronize();

    for (int i = 0; i < trial_runs; ++i) {
        cudaEventRecord(start, 0);

        Amtal::run_cublas_sgemm_bf16(device_a, device_b, device_c, M, N, K, alpha, beta);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float run_time;
        cudaEventElapsedTime(&run_time, start, stop);
        elapsed_time += run_time;
    }

    elapsed_time /= trial_runs;
    elapsed_time /= 1000.0f; // convert to seconds;
    
    long long FLOPS = (2LL * K * M * N) / elapsed_time / 1e9;

    std::cout << "Total GFLOPS " << FLOPS << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(host_a);
    free(host_b);
    free(host_c);
    free(copy_host_c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(copy_device_c);

    cublasDestroy(cublas_handle);
    return 0;
}
