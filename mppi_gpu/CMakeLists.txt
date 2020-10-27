cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)
include_directories(include)

add_executable(mppi_gpu src/main.cu src/point_mass.cu)
set_target_properties(mppi_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
