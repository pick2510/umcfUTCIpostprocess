cmake_minimum_required(VERSION 3.16)
project(umcfUTCIpostprocess CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCES
    umcfUTCIpostprocess.C
    constants.H
    types.H
    raycaster.C
    viewFactor.C
    tmrtSolver.C
    utciSolver.C
    pedestrian.C
    io.C
    caching.C
)

add_executable(umcfUTCIpostprocess ${SOURCES})

target_link_libraries(umcfUTCIpostprocess
    PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

install(TARGETS umcfUTCIpostprocess DESTINATION bin)
