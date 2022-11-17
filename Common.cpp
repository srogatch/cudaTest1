#include "Common.h"

CudaAttributes::CudaAttributes(const int i_gpu) {
  cudaDeviceProp prop;
  gpuErrchk(cudaGetDeviceProperties(&prop, i_gpu));
  max_parallelism_ = uint32_t(prop.maxThreadsPerMultiProcessor) * prop.multiProcessorCount;
  gpuErrchk(cudaSetDevice(i_gpu));

  size_t free_bytes, total_bytes;
  gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));
  // Enable mergesort, that allocates device memory
  gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, free_bytes>>1));

  gpuErrchk(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost | cudaDeviceLmemResizeToMax));
}
