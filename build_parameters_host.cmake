# include guard
if(BUILD_PARAMETERS_HOST_INCLUDED)
    return()
endif(BUILD_PARAMETERS_HOST_INCLUDED)

# save culip root dir
set(CULIP_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# C/C++ compiler
set(CMAKE_CXX_STANDARD 11)

# general host compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -march=native -m64 -DGPU_BLAS -Wfatal-errors")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g -O3")
else(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif(CMAKE_BUILD_TYPE MATCHES Debug)


# mark file as processed
SET(BUILD_PARAMETERS_HOST_INCLUDED TRUE)