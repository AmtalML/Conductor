#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

// Error checking Macro
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while(0)

// Ceiling Division Macro
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

