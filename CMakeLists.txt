# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(Conductor LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads REQUIRED)
add_library(engine STATIC
    engine/src/inference_engine_impl.cu
)
target_include_directories(engine PUBLIC
    engine/include
)
add_executable(llm_server
    server/main.cpp
)
target_include_directories(llm_server PRIVATE
    ${Boost_INCLUDE_DIRS}
    engine/include
)
target_link_libraries(llm_server PRIVATE
    engine
    ${Boost_LIBRARIES}
    Threads::Threads
)
