cmake_minimum_required(VERSION 3.23)
project(rsi-user-radar LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/cluster.cu
    src/radar.cu
)

set(VENDOR_SDK_ROOT "" CACHE PATH "Path to local vendor SDK")

if (NOT VENDOR_SDK_ROOT)
    message(FATAL_ERROR "Set -DVENDOR_SDK_ROOT=... to your local SDK path.")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${VENDOR_SDK_ROOT}/include
    ${VENDOR_SDK_ROOT}/samples/common
    include
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CMAKE_C_VISIBILITY_PRESET hidden
    CMAKE_CXX_VISIBILITY_PRESET hidden
    CMAKE_CUDA_VISIBILITY_PRESET hidden
    CMAKE_VISIBILITY_INLINES_HIDDEN 1
)
