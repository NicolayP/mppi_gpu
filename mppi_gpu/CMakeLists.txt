cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cmake_and_cuda LANGUAGES CXX CUDA)
include_directories(include include/mujoco/)

set(USE_GL 1)

set(BIN_NAME mppi_gpu)

# Set c11 standard
set(CMAKE_CXX_STANDARD 11)

# N.B. This is a temporary flag. It only really applies to Linux, as Mac
# does not need X11.
option(RUN_X11_TESTS "Run tests that require X11" OFF)

#Mujoco paths
set(MUJOCO_MJPRO_PATH ${CMAKE_CURRENT_SOURCE_DIR})

#Paths
link_directories(lib lib/contrib/lib/)

#Showing path to MuJoCo for checking
message(STATUS "MuJoCo path: " ${MUJOCO_MJPRO_PATH})

#Finding main mujoco library
if(${USE_GL})
    file(GLOB LIB_MUJOCO ${MUJOCO_MJPRO_PATH}/lib/contrib/lib/libmujoco[0-9][0-9][0-9].so)
else()
    file(GLOB LIB_MUJOCO ${MUJOCO_MJPRO_PATH}/lib/contrib/lib/libmujoco[0-9][0-9][0-9]nogl.so)
endif()

#Showing mujoco library found
message(STATUS "MuJoCo lib: " ${LIB_MUJOCO})

find_package(glfw3 3.4 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRECTORIES})
message(STATUS "GLFW path: " ${GLFW_INCULUDE_DIRECTORIES})

find_package(GLEW 2.0 REQUIRED)

include_directories(${GLEW_INCULUDE_DIRECTORIES})
message(STATUS "GLEW path: " ${GLEW_INCULUDE_DIRECTORIES})


add_executable(${BIN_NAME} src/main.cu src/point_mass.cu src/point_mass_gpu.cu src/mppi_utils.cu src/cost.cu src/PointMassEnv.cpp src/Env.cpp)

set_property(TARGET ${BIN_NAME} PROPERTY CUDA_ARCHITECTURES 70-real)

target_link_options(${BIN_NAME} PRIVATE $<DEVICE_LINK:-arch=sm_70>)

target_compile_options(${BIN_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -g -G -arch sm_70>)

set_target_properties(${BIN_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})

target_link_libraries(${BIN_NAME} GL GLU glut)

target_link_libraries(${BIN_NAME} ${GLEW_LIBRARIES} glfw)
