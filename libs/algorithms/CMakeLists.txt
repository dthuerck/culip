file(GLOB_RECURSE LIBS_ALGORITHMS_CPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/algorithms/*.cc")
file(GLOB_RECURSE LIBS_ALGORITHMS_GPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/algorithms/*.cu")

cuda_add_library(libalgorithms
    ${LIBS_ALGORITHMS_CPU_SOURCE}
    ${LIBS_ALGORITHMS_GPU_SOURCE}
    SHARED)

set(LIBS_ALGORITHMS
    libalgorithms)



