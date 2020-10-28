cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)
include_directories(include)

add_executable(gpu_class src/main.cu src/gpu_class.cu)

set_property(TARGET gpu_class PROPERTY CUDA_ARCHITECTURES 70-real)
target_link_options(gpu_class PRIVATE $<DEVICE_LINK:-arch=sm_70>)

target_compile_options(gpu_class PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo -arch sm_70>)
set_target_properties(gpu_class PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
