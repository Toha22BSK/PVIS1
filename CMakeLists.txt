cmake_minimum_required(VERSION 3.0)
project(1)

set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_executable(1 main.cpp utils.cpp pch.h utils.h GPUFunction.h)

target_compile_definitions(1
        PRIVATE
        CL_HPP_ENABLE_EXCEPTIONS
        CL_HPP_TARGET_OPENCL_VERSION=200
        )

target_link_libraries(1 PRIVATE CONAN_PKG::khronos-opencl-clhpp)
