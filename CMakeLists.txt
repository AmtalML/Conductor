cmake_minimum_required(VERSION 3.18)
project(Conductor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

include(FetchContent)
FetchContent_Declare(
    Boost
    URL https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2
)

find_package(BOOST REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CUDA_INCLUDE ${CUDA_TOOLKIT_ROOT_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE})

if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found")
endif()

add_executable(batching_service server/batching_service.cpp)
add_executable(conductor_server server/main.cpp)


target_link_libraries(Conductor PRIVATE Boost::boost)