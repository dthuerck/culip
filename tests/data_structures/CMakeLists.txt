if(BUILD_TESTS_DATA_STRUCTURES)
    file(GLOB SOURCE_TESTS_DATA_STRUCTURES "${PROJECT_SOURCE_DIR}/tests/data_structures/*.cc")

    cuda_add_executable(culip-tests-datastructures
        ${SOURCE_TESTS_DATA_STRUCTURES})
    add_dependencies(culip-tests-datastructures googletest)
    target_link_libraries(culip-tests-datastructures
        ${LIBS_UTILS}
        ${LIBS_DATA_STRUCTURES}
        ${LIBS_ALGORITHMS}
        ${LIBS_TEST}
        ${LIBS_LA}
        ${DEPENDENCY_GTEST}
        ${DEVICE_LIBRARIES})
endif(BUILD_TESTS_DATA_STRUCTURES)
