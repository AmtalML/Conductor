#include "../Conductor/include/utils.cuh"
#include <cstdlib>


namespace Amtal {


__global__ void warmupKernel() {
    __shared__ float s[100];
    s[0] += s[1];
}

}




int main() {

    // Profiling SGEMM Kernel Flops 

    int N = 1 >> 10;

    float *A, *B, *C;

    A = malloc(sizeof(float) * N * N), B = malloc(sizeof(float) * N * N);

    
    // warmup kernel
    warmupKernel<<<1024, 1024>>>();

    cublasCreate()



    return 0;

}