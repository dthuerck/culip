if(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cub/include)
    file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/dependencies/cub/include)
endif(NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cub/include)

# copy directly from the git repo
externalproject_add(cub
    PREFIX ext_cub
    # download step
    GIT_REPOSITORY https://github.com/NVlabs/cub.git
    GIT_TAG master
    UPDATE_COMMAND ""
    # header-only library, just copy files
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND cp -R cub ${PROJECT_SOURCE_DIR}/dependencies/cub/include
    INSTALL_COMMAND "")

set(INCLUDE_CUB
    ${PROJECT_SOURCE_DIR}/dependencies/cub/include)
