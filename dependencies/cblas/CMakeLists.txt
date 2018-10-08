if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cblas/include)
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/dependencies/cblas/include)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cblas/include)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cblas/include/cblas.h)
    message("Downloading cblas...")
    file(DOWNLOAD http://www.netlib.org/blas/cblas.h ${PROJECT_SOURCE_DIR}/dependencies/cblas/include/cblas.h)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cblas/include/cblas.h)

set(INCLUDE_CBLAS
    ${PROJECT_SOURCE_DIR}/dependencies/cblas/include)
set(DEPENDENCY_CBLAS
    blas)