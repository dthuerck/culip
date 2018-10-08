file(GLOB LIBS_TEST_CPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/test/*.cc")
file(GLOB LIBS_TEST_GPU_SOURCE "${PROJECT_SOURCE_DIR}/libs/test/*.cu")

cuda_add_library(libtest
    ${LIBS_TEST_CPU_SOURCE}
    ${LIBS_TEST_GPU_SOURCE}
    SHARED)
add_dependencies(libtest
    mmio)

set(LIBS_TEST
    ${DEPENDENCY_MMIO}
    libtest)