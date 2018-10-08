# create folder for test data
if(NOT EXISTS ${PROJECT_BINARY_DIR}/data)
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/data)
endif(NOT EXISTS ${PROJECT_BINARY_DIR}/data)

include(${PROJECT_SOURCE_DIR}/tests/algorithms/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/tests/data_structures/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/tests/la/CMakeLists.txt)
