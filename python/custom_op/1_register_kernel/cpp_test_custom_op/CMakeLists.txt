cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 17)

set(TARGET_NAME "test_custom_op_cpp_app")

project(${TARGET_NAME})

find_package(OpenVINO REQUIRED)

# 添加你的源代码
add_executable(${TARGET_NAME} main.cpp)

# 链接 OpenVINO 库
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)