cmake_minimum_required(VERSION 3.13)

set(PRJ_NAME testapp)
project(${PRJ_NAME})

find_package(OpenVINO REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# OneTBB
include_directories(./)

add_executable(${PRJ_NAME} 
    main.cpp
    model_multiply.cpp
    model_conv_bias_sum_reshape.cpp
    compare_cpu_template.cpp
)

target_link_libraries(${PRJ_NAME} 
    openvino::runtime
)