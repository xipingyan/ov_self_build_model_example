cmake_minimum_required(VERSION 3.13)

set(PRJ_NAME testapp)
project(${PRJ_NAME})

find_package(OpenVINO REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# OneTBB
include_directories(./header/)
file(GLOB SRC "header/*.h" "header/*.hpp" "src/*.cpp")

add_executable(${PRJ_NAME} ${SRC})

target_link_libraries(${PRJ_NAME} 
    openvino::runtime
)