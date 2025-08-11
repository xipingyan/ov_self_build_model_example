cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 17)

set(TARGET_NAME "openvino_custom_add_extension")
project($TARGET_NAME)

# The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    message(STATUS, Python3_Interpreter_FOUND=${Python3_Interpreter_FOUND})
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
            OUTPUT_VARIABLE OpenVINO_DIR_PY
            ERROR_QUIET)
    endif()
endif()

message(status "${OpenVINO_DIR_PY}")
find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

set(SRC custom_add.cpp ov_extension.cpp custom_eltwise.cpp custom_add_2_output.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)