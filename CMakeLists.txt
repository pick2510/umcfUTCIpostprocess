cmake_minimum_required(VERSION 3.16)
project(umcfUTCIpostprocess CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

find_package(OpenMP REQUIRED)

# Use bundled Eigen from extern/eigen submodule (header-only, no install needed)
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen")
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen headers not found at ${EIGEN3_INCLUDE_DIR}.\n"
        "Run:  git submodule update --init --recursive")
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")

set(SOURCES
    src/umcfUTCIpostprocess.cpp
    src/raycaster.cpp
    src/viewFactor.cpp
    src/tmrtSolver.cpp
    src/utciSolver.cpp
    src/pedestrian.cpp
    src/io.cpp
    src/caching.cpp
)

add_executable(umcfUTCIpostprocess ${SOURCES})

target_include_directories(umcfUTCIpostprocess PRIVATE src ${EIGEN3_INCLUDE_DIR})

target_link_libraries(umcfUTCIpostprocess
    PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

install(TARGETS umcfUTCIpostprocess DESTINATION bin)
