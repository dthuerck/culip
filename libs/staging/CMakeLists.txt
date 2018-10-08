file(GLOB_RECURSE LIBS_STAGING_CPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/staging/*.cc")
file(GLOB_RECURSE LIBS_STAGING_GPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/staging/*.cu")

cuda_add_library(libstaging
    ${LIBS_STAGING_CPU_SOURCE}
    ${LIBS_STAGING_GPU_SOURCE}
    SHARED)

set(LIBS_STAGING
    libstaging)






