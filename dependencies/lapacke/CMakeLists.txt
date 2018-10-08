if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include)
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include/lapacke.h)
    message("Downloading lapacke...")
    file(DOWNLOAD http://www.netlib.org/lapack/lapacke.h ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include/lapacke.h)
    file(DOWNLOAD http://www.netlib.org/lapack/lapacke_mangling.h ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include/lapacke_mangling.h)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include/lapacke.h)

set(INCLUDE_LAPACKE
    ${PROJECT_SOURCE_DIR}/dependencies/lapacke/include)
set(DEPENDENCY_LAPACKE
    lapacke)