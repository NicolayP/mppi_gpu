#ifndef __GPU_CMAKE_H__
#define __GPU_CMAKE_H__

#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000000
#define ERR_TOL 1e-6

__global__ void kernel(float* out, float* a, float* b, int n);

#endif
