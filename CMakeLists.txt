cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(cudaTest1_proj LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include(CTest)

set(CMAKE_CUDA_ARCHITECTURES "61;75;86" CACHE STRING "CUDA compute capabilities to target for")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount 32")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options --suppress-stack-size-warning")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_executable(cudaTest1 main.cpp kernel.cu Common.cpp)
set_property(TARGET cudaTest1 PROPERTY CUDA_STANDARD 20)

# lots of warnings and all warnings as errors
add_compile_options(-Wall -Wextra -pedantic -Werror -march=native)

# Use parallel STL
find_package(OpenMP)
if (OPENMP_FOUND)
  target_link_libraries(cudaTest1 PRIVATE OpenMP::OpenMP_CXX)
  add_compile_definitions(_GLIBCXX_PARALLEL)
endif()
