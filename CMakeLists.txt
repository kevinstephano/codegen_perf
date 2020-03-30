cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(codegen_perf)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(codegen_perf codegen_perf.cpp)
target_link_libraries(codegen_perf "${TORCH_LIBRARIES}")
set_property(TARGET codegen_perf PROPERTY CXX_STANDARD 14)
