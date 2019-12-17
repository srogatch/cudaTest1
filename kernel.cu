#include "cuda_interop.h"
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

__global__ void TestKernel() {
    printf("Hello CUDA!\n");
}

void TestOnCuda() {
    cudaError_t err = cudaSetDevice(0);
    if(err != 0) {
        fprintf(stderr, "cudaSetDevice() returned %d: %s\n", err, cudaGetErrorString(err));
    }
    TestKernel<<<1,1>>>();
    err = cudaGetLastError();
    if(err != 0) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", err, cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if(err != 0) {
        fprintf(stderr, "cudaDeviceSynchronize() returned %d: %s\n", err, cudaGetErrorString(err));
    }
}