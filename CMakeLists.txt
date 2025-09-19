cmake_minimum_required(VERSION 3.16)

project(basic_interop_skel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Prefer NEW behavior for archive timestamps when using FetchContent (CMP0135)
if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Set CUDA architectures (RTX 3060 = Ampere SM 86) and adopt CMP0104
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

# GLFW: prefer vendored copy if present; otherwise fetch a tagged release
# GPT GENERATED: uses FetchContent to download GLFW 3.4 if not downloaded locally in external/glfw
include(FetchContent)
if (EXISTS "${CMAKE_SOURCE_DIR}/external/glfw/CMakeLists.txt")
    message(STATUS "Using vendored GLFW at external/glfw")
    add_subdirectory(external/glfw)
else()
    message(STATUS "Fetching GLFW (3.4) via FetchContent")
    FetchContent_Declare(
        glfw
        URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(glfw)
endif()

add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

enable_language(CUDA)
add_compile_definitions(HAS_CUDA=1)

set(APP_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/callbacks.cpp
    ${SRC_DIR}/cuda_kernels.cu
    ${SRC_DIR}/shader_utils.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${APP_SOURCES})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/external/glad/include"
    "${CMAKE_SOURCE_DIR}/external/glfw/include"
    ${INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
                      glad
                      glfw
                      cuda
                      cudart_static
                      )

# Copy shader files to build directory (next to executable) for runtime loading
file(GLOB SHADER_FILES "${CMAKE_SOURCE_DIR}/shaders/*.*")
if (SHADER_FILES)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_FILES} "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/shaders"
        COMMENT "Copying shader files")
endif()