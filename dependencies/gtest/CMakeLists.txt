if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib)
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib)

externalproject_add(googletest
    PREFIX ${PROJECT_SOURCE_DIR}
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG master
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=ON
    UPDATE_COMMAND ""
    INSTALL_COMMAND pwd COMMAND cp lib/libgtest.so ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib/ COMMAND cp lib/libgtest_main.so ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib/ COMMAND cp -R ../googletest/googletest/include ${PROJECT_SOURCE_DIR}/dependencies/gtest/)

# save the library location for later
set(INCLUDE_GTEST
    ${PROJECT_SOURCE_DIR}/dependencies/gtest/include)
set(DEPENDENCY_GTEST
    ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib/libgtest.so
    ${PROJECT_SOURCE_DIR}/dependencies/gtest/lib/libgtest_main.so)
