#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr const uint32_t kThreadBlockSize = 256;

// CUDA error checking helper macro.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// Check that the CUDA call is successful, and if not, exit the application.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU error %d: %s . %s:%d\n", int(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct CudaAttributes {
  uint32_t max_parallelism_;
  explicit CudaAttributes(const int i_gpu);
};
