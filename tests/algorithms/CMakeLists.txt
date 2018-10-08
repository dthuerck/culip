if(BUILD_TESTS_ALGORITHMS)
    file(GLOB SOURCE_TESTS_ALGORITHMS "${PROJECT_SOURCE_DIR}/tests/algorithms/*.cc")

    cuda_add_executable(culip-tests-algorithms
        ${SOURCE_TESTS_ALGORITHMS})
    add_dependencies(culip-tests-algorithms googletest)
    target_link_libraries(culip-tests-algorithms
        ${LIBS_UTILS}
        ${LIBS_DATA_STRUCTURES}
        ${LIBS_ALGORITHMS}
        ${LIBS_TEST}
        ${LIBS_LA}
        ${DEPENDENCY_GTEST}
        ${DEPENDENCY_SPRAL}
        ${DEPENDENCY_METIS}
	    ${DEPENDENCY_MMIO}
	    ${DEPENDENCY_OPENBLAS}
        ${DEPENDENCY_CBLAS}
	    ${DEPENDENCY_LAPACKE}
        ${DEVICE_LIBRARIES})
    add_custom_command(TARGET culip-tests-algorithms
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/algorithms/data/* ${PROJECT_BINARY_DIR}/data)
endif(BUILD_TESTS_ALGORITHMS)
