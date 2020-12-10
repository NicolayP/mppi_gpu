cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
find_package(CUDA REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "-std=c++11  -fext-numeric-literals")
  #'-fext-numeric-literals' : https://svn.boost.org/trac/boost/ticket/9240
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CUDA_COMPILER  /usr/local/cuda-10.0/bin/nvcc)
project(mppi_gpu LANGUAGES CXX CUDA)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  uuv_auv_control_allocator
)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include 
  include/mujoco/
  ${catkin_INCLUDE_DIRS}
)

set(USE_GL 1)

set(BIN_NAME mppi_gpu)
set(BIN_LIB ${BIN_NAME}_lib)
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

find_package(glfw3 3.1 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRECTORIES})
message(STATUS "GLFW path: " ${GLFW_INCULUDE_DIRECTORIES})

find_package(GLEW 2.0 REQUIRED)
find_package(PkgConfig)

pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)
include_directories(${YAMLCPP_INCLUDE_DIRS})

include_directories(${GLEW_INCULUDE_DIRECTORIES})
message(STATUS "GLEW path: " ${GLEW_INCULUDE_DIRECTORIES})


# add_executable(${BIN_NAME} src/main.cu src/point_mass.cu src/point_mass_gpu.cu src/mppi_utils.cu src/cost.cu src/PointMassEnv.cpp src/Env.cpp)
SET(CUDA_NVCC_FLAGS "-arch=sm_70 --relocatable-device-code true" CACHE STRING "nvcc flags" FORCE)
# SET(CUDA_NVCC_FLAGS "-arch=sm_70" CACHE STRING "nvcc flags" FORCE)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 
CUDA_ADD_LIBRARY(${BIN_LIB}
  ${LIB_TYPE}
  src/point_mass.cu
  src/point_mass_gpu.cu
  src/cost.cu
  src/mppi_utils.cu
  
)

ADD_EXECUTABLE(node 
  src/main.cpp
  src/PointMassEnv.cpp
  src/Env.cpp
)

ADD_DEPENDENCIES(node 
  ${BIN_LIB}
  ${catkin_EXPORTED_TARGETS}
)

target_link_options(node PRIVATE $<DEVICE_LINK:-arch=sm_70>)
set_target_properties(node PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

TARGET_LINK_LIBRARIES(node
  ${catkin_LIBRARIES}
  ${LIB_MUJOCO}
  GL GLU glut
  ${GLEW_LIBRARIES} glfw
  ${PCL_LIBRARIES}
  ${YAMLCPP_LIBRARIES}
  ${BIN_LIB}
)

# set_property(TARGET ${BIN_NAME} PROPERTY CUDA_ARCHITECTURES 70-real)

# target_link_options(${BIN_NAME} PRIVATE $<DEVICE_LINK:-arch=sm_70>)

# target_compile_options(${BIN_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic -g -G -arch sm_70>)

# set_target_properties(${BIN_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})

# target_link_libraries(${BIN_NAME} GL GLU glut)

# target_link_libraries(${BIN_NAME} ${GLEW_LIBRARIES} glfw)

# target_link_libraries(${BIN_NAME} ${catkin_LIBRARIES}  ${YAMLCPP_LIBRARIES})
