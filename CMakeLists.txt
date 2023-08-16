cmake_minimum_required(VERSION 3.26)
project(PRUEBA_PARCIAL)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#### FMT
include(FetchContent)

FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

add_executable(PRUEBA_PARCIAL main.cpp)

target_link_libraries(PRUEBA_PARCIAL fmt::fmt)
