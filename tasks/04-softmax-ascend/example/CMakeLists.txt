cmake_minimum_required(VERSION 3.20)

set(target_suffix example CACHE STRING "Suffix appended to the softmax_ascend executable name")
set(target_name "softmax_ascend_${target_suffix}")
set(kernel_target "${target_name}_kernels")

set(SOC_VERSION "Ascend910B1" CACHE STRING "System on Chip type used for compilation")

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
    elseif(DEFINED ENV{ASCEND_INSTALL_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_INSTALL_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
    else()
        set(_ascend_default_path "$ENV{HOME}/Ascend/ascend-toolkit/latest")
        if(NOT EXISTS "${_ascend_default_path}" AND EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
            set(_ascend_default_path "/usr/local/Ascend/ascend-toolkit/latest")
        endif()
        set(ASCEND_CANN_PACKAGE_PATH "${_ascend_default_path}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
        unset(_ascend_default_path)
    endif()
else()
    set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(ascendc_kernel_found OFF)
foreach(candidate_dir IN ITEMS
        "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake"
        "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
        "${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake")
    if(EXISTS "${candidate_dir}")
        set(ASCENDC_CMAKE_DIR "${candidate_dir}")
        set(ascendc_kernel_found ON)
        break()
    endif()
endforeach()

if(ascendc_kernel_found)
    include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
    ascendc_library(${kernel_target} STATIC
        softmax_kernel.cpp
    )
else()
    message(FATAL_ERROR
        "Unable to locate ascendc_kernel_cmake under '${ASCEND_CANN_PACKAGE_PATH}'. "
        "Set ASCEND_CANN_PACKAGE_PATH to a valid CANN installation to enable AscendC builds.")
endif()

add_executable(${target_name}
    main.cpp
)

target_compile_features(${target_name} PRIVATE cxx_std_17)

target_include_directories(${target_name} PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include/acl"
)

target_link_libraries(${target_name} PRIVATE
    ${kernel_target}
)
