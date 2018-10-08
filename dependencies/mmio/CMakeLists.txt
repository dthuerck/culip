if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/mmio/include)
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/dependencies/mmio/include)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/mmio/include)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/mmio/lib)
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/dependencies/mmio/lib)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/mmio/lib)

externalproject_add(mmio
    PREFIX ext_mmio
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/dependencies/mmio/src
    DOWNLOAD_COMMAND rm -f ${PROJECT_SOURCE_DIR}/dependencies/mmio/src/mmio.c COMMAND rm -f ${PROJECT_SOURCE_DIR}/dependencies/mmio/src/mmio.h COMMAND wget -nc http://math.nist.gov/MatrixMarket/mmio/c/mmio.c -O ${PROJECT_SOURCE_DIR}/dependencies/mmio/src/mmio.c COMMAND wget -nc http://math.nist.gov/MatrixMarket/mmio/c/mmio.h -O ${PROJECT_SOURCE_DIR}/dependencies/mmio/src/mmio.h
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_C_COMPILER} -c -Wall -fpic mmio.c COMMAND ${CMAKE_C_COMPILER} -shared -o libmmio.so mmio.o
    INSTALL_COMMAND cp libmmio.so ${PROJECT_SOURCE_DIR}/dependencies/mmio/lib/ COMMAND cp mmio.h ${PROJECT_SOURCE_DIR}/dependencies/mmio/include)

# save the library location for later
set(INCLUDE_MMIO
    ${PROJECT_SOURCE_DIR}/dependencies/mmio/include)
set(DEPENDENCY_MMIO
    ${PROJECT_SOURCE_DIR}/dependencies/mmio/lib/libmmio.so)

