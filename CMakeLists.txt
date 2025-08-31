cmake_minimum_required(VERSION 3.10)
project(stiefel_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Path to Eigen3 headers")

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/include   
)

add_executable(run_stiefel
  main.cpp
  src/StiefelProduct.cpp
  src/ScaledStiefelProduct.cpp
)
