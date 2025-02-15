cmake_minimum_required(VERSION 3.18)
project(Conductor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB_RECURSE LIB_SOURCES
  "${PROJECT_SOURCE_DIR}/Conductor/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/Conductor/src/*.cu"
)

add_library(Conductor ${LIB_SOURCES})
target_include_directories(Conductor PUBLIC ${PROJECT_SOURCE_DIR}/Conductor/include)


add_executable(Conductor src/main.cpp)
target_link_libraries(Conductor PRIVATE Conductor)

enable_testing()
add_subdirectory(tests)

