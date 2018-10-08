# use external sources conveniently
include(ExternalProject)

# download and/or build external dependencies
include(${PROJECT_SOURCE_DIR}/dependencies/cblas/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/dependencies/cub/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/dependencies/gtest/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/dependencies/lapacke/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/dependencies/mmio/CMakeLists.txt)

# use system blas
set(DEPENDENCY_OPENBLAS
    blas
    hwloc
    gfortran)

# set includes
set(INCLUDE_DEPENDENCIES
    ${INCLUDE_GTEST}
    ${INCLUDE_CBLAS}
    ${INCLUDE_CUB}
    ${INCLUDE_LAPACKE}
    ${INCLUDE_OPENBLAS}
    ${INCLUDE_MMIO})

