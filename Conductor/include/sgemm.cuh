#include "utils.cuh"


namespace Amtal {

__global__ void naive_sgemm(bf16 *A, bf16 *B, bf16 *C, int N, int M, int K);



}