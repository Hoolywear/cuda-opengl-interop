cmake_minimum_required(VERSION 3.16)

project(basic_interop_skel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Prefer NEW behavior for archive timestamps when using FetchContent (CMP0135)
if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

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

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
                            "${CMAKE_SOURCE_DIR}/external/glad/include"
                            "${CMAKE_SOURCE_DIR}/external/glfw/include"
                            )

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
                      glad
                      glfw
                      )