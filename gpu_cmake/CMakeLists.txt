cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)

include_directories(include)

add_executable(gpu src/gpu_cmake.cu)
set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
