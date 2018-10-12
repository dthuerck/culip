# include guard
if(BUILD_PARAMETERS_DEVICE_INCLUDED)
    return()
endif(BUILD_PARAMETERS_DEVICE_INCLUDED)

# general CUDA options
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_SEPARABLE_COMPILATION ON)
# note: use semicolon to list multiple architectures
set(CUDA_CC "35;50;52;60;61;70")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

# force using CUDA package
find_package(CUDA REQUIRED)

# debug / release flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-Xcompiler -std=c++11;-rdc=true;-Xcompiler -fPIC;-Xcompiler -fopenmp;-prec-div=true;-prec-sqrt=true)
foreach(arch ${CUDA_CC})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_${arch},code=sm_${arch})
endforeach(arch)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G;-g)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-pg;-g;-lineinfo)
else(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-lineinfo)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

# include culip tree root directory
include_directories("${CMAKE_CURRENT_LIST_DIR}")

# include SDK directory
set(CUDA_SDK_ROOT_DIR CACHE STRING "/usr/local/cuda/samples")
include_directories("${CUDA_SDK_ROOT_DIR}/Common" SYSTEM)
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include/crt" SYSTEM)

# create a list of CUDA libs to link against
set(DEVICE_LIBRARIES
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusolver.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudadevrt.a)

# mark file as processed
set(BUILD_PARAMETERS_DEVICE_INCLUDED TRUE)
