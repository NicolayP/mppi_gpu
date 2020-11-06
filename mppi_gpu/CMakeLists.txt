cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)
include_directories(include)

add_executable(mppi_gpu src/main.cu src/point_mass.cu src/cost.cu)

set_property(TARGET mppi_gpu PROPERTY CUDA_ARCHITECTURES 70-real)

target_link_options(mppi_gpu PRIVATE $<DEVICE_LINK:-arch=sm_70>)

target_compile_options(mppi_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -g -G -arch sm_70>)

set_target_properties(mppi_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
