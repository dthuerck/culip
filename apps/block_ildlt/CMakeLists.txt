if(BUILD_APP_BLOCK_ILDLT)
    cuda_add_executable(culip-block-ildlt
        ${PROJECT_SOURCE_DIR}/apps/block_ildlt/block_ildlt.cc)
    target_link_libraries(culip-block-ildlt
        ${LIBS_UTILS}
        ${LIBS_DATA_STRUCTURES}
        ${LIBS_TEST}
        ${LIBS_LA}
        ${LIBS_ALGORITHMS}
        ${DEPENDENCY_CBLAS}
        ${DEPENDENCY_LAPACKE}
        ${DEPENDENCY_SPRAL}
        ${DEPENDENCY_METIS}
        ${DEPENDENCY_MMIO}
	${DEPENDENCY_OPENBLAS}
        ${DEVICE_LIBRARIES})
endif(BUILD_APP_BLOCK_ILDLT)

