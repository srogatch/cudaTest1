cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cudaTest1 LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=32 -gencode=arch=compute_61,code=\\\"sm_61,compute_61\\\"")

add_executable(cudaTest1 main.cpp kernel.cu cuda_interop.h)