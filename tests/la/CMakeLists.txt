if(BUILD_TESTS_LA)
    file(GLOB SOURCE_TESTS_LA "${PROJECT_SOURCE_DIR}/tests/la/*.cc")

    cuda_add_executable(culip-tests-la
        ${SOURCE_TESTS_LA})
    add_dependencies(culip-tests-la googletest)
    set_property(TARGET culip-tests-la PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(culip-tests-la
        ${LIBS_UTILS}
        ${LIBS_DATA_STRUCTURES}
        ${LIBS_LA}
        ${LIBS_ALGORITHMS}
        ${LIBS_TEST}
        ${DEPENDENCY_OPENBLAS}
        ${DEPENDENCY_CBLAS}
        ${DEPENDENCY_LAPACKE}
        ${DEPENDENCY_SPRAL}
        ${DEPENDENCY_METIS}
	    ${DEPENDENCY_MMIO}
        ${DEPENDENCY_GTEST}
	    ${DEVICE_LIBRARIES})
    add_custom_command(TARGET culip-tests-la POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/la/data/* ${PROJECT_BINARY_DIR}/data)
endif(BUILD_TESTS_LA)
