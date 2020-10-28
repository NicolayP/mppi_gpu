cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)
include_directories(include)

add_executable(gpu_class src/main.cu src/gpu_class.cu)
target_compile_options(gpu_class PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo -arch sm_70>)
set_target_properties(gpu_class PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
