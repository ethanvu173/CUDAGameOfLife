cmake_minimum_required(VERSION 3.18)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

project(GameOfLife LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

add_executable(game_of_life
    src/main.cpp
    src/simulation.cu
)

set_target_properties(game_of_life PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(game_of_life PRIVATE src/)
target_link_libraries(game_of_life PRIVATE
    SFML::Graphics SFML::Window SFML::System
)