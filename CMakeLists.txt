cmake_minimum_required(VERSION 3.18)
project(Conductor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB_RECURSE LIB_SOURCES
  "${CMAKE_SOURCE_DIR}/Conductor/src/*.cpp"
  "${CMAKE_SOURCE_DIR}/Conductor/src/*.cu"
)

add_library(Conductor ${LIB_SOURCES})
target_include_directories(Conductor PUBLIC ${CMAKE_SOURCE_DIR}/Conductor/include)

add_executable(ConductorExec Conductor/src/main.cu)
add_executable(SGEMM_PERF profiling/profiling_sgemm.cu)

target_link_libraries(ConductorExec PRIVATE Conductor)
target_link_libraries(SGEMM_PERF PRIVATE Conductor)
