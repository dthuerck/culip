file(GLOB LIBS_DATA_STRUCTURES_CPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/data_structures/*.cc")
file(GLOB LIBS_DATA_STRUCTURES_GPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/data_structures/*.cu")

cuda_add_library(libdatastructures
    ${LIBS_DATA_STRUCTURES_CPU_SOURCE}
    ${LIBS_DATA_STRUCTURES_GPU_SOURCE}
    SHARED)

set(LIBS_DATA_STRUCTURES
    libdatastructures)


