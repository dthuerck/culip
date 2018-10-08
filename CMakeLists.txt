# Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-clause license. See the LICENSE file for details.
cmake_minimum_required(VERSION 3.2)

# start project
project(culip)

#include host build parameters
include(${CMAKE_CURRENT_LIST_DIR}/build_parameters_host.cmake)

# default build type
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type.")

# set build parameters
set(BUILD_APP_BLOCK_ILDLT ON CACHE BOOL "Build the Block-iLDLt app.")
set(BUILD_APP_BLOCKING_STATS ON CACHE BOOL "Build the blocking-stats app.")
set(BUILD_TESTS_ALGORITHMS ON CACHE BOOL "Build tests for libalgorithms.")
set(BUILD_TESTS_DATA_STRUCTURES ON CACHE BOOL "Build tests for libdatastructures.")
set(BUILD_TESTS_LA ON CACHE BOOL "Build tests for libla.")
set(BUILD_TESTS_STAGING ON CACHE BOOL "Build tests for staging code.")

# include CUDA build parameters
include(${CMAKE_CURRENT_LIST_DIR}/build_parameters_device.cmake)

# add dependencies
include(${PROJECT_SOURCE_DIR}/dependencies/CMakeLists.txt)
include_directories(${INCLUDE_DEPENDENCIES})

# build libraries
include(${PROJECT_SOURCE_DIR}/libs/CMakeLists.txt)

# build selected apps
include(${PROJECT_SOURCE_DIR}/apps/CMakeLists.txt)

# build selected tests
include(${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt)
