cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)


include_directories(include)

add_executable(gpu src/gpu_cmake.cu)

set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 70-real)

target_link_options(gpu PRIVATE $<DEVICE_LINK:-arch=sm_70>)

target_compile_options(gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -lineinfo -arch sm_70>)

set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
