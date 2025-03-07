cmake_minimum_required(VERSION 3.18)
project(Conductor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(Boost REQUIRED COMPONENTS filesystem system)

if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found")
endif()

add_executable(batching_service server/batching_service.cpp)
add_executable(conductor_server server/main.cpp)

target_link_libraries(batching_service PRIVATE Boost::filesystem Boost::system)
target_link_libraries(conductor_server PRIVATE Boost::filesystem Boost::system)
