cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(cudaTest1_proj LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include(CTest)

set(CMAKE_CUDA_ARCHITECTURES 61 75 86 CACHE STRING "CUDA compute capabilities to target for" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount 40 --use_fast_math")
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options --suppress-stack-size-warning")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# lots of warnings and all warnings as errors
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra \
    -Xcompiler -mavx2 -Xcompiler -mfma -Xcompiler -mrdrnd")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror -mavx2 -mfma -mrdrnd -flto")
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror -mavx2 -mfma -mrdrnd -flto")
elseif(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4 -Xcompiler /arch:AVX2 \
    -Xcompiler /GL -Xcompiler /Qvec -Xcompiler /Oi -Xcompiler /Ob3 -Xcompiler /WX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2 /GL /Qvec /Oi /Ob3 /WX")
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2 /GL /Qvec /Oi /Ob3 /WX")
else()
  message(FATAL_ERROR "Unsupported compiler")
endif()

add_executable(cudaTest1 main.cpp kernel.cu Common.cpp)
target_link_libraries(cudaTest1 PRIVATE CUDA::cudart)

# Use parallel STL
find_package(OpenMP)
if (OPENMP_FOUND)
  target_link_libraries(cudaTest1 PRIVATE OpenMP::OpenMP_CXX)
  if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_definitions(_GLIBCXX_PARALLEL)
  endif()
endif()
