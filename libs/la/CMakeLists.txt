file(GLOB_RECURSE LIBS_LA_CPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/la/*.cc")
file(GLOB_RECURSE LIBS_LA_GPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/la/*.cu")

cuda_add_library(libla
    ${LIBS_LA_CPU_SOURCE}
    ${LIBS_LA_GPU_SOURCE}
    SHARED)
add_dependencies(libla
    cub)

set(LIBS_LA
    libla)






