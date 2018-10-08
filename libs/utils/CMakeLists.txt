file(GLOB LIBS_UTILS_CPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/utils/*.cc")
file(GLOB LIBS_UTILS_GPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/utils/*.cu")

cuda_add_library(libutils
    ${LIBS_UTILS_CPU_SOURCE}
    ${LIBS_UTILS_GPU_SOURCE}
    SHARED)

set(LIBS_UTILS
    libutils)


