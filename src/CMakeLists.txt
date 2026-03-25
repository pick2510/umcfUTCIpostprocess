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

set(VTK_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu")

set(VTK_LIBRARIES
    ${VTK_LIBRARY_DIR}/libvtkCommonCore-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkCommonDataModel-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkCommonExecutionModel-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkFiltersCore-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkFiltersExtraction-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkFiltersGeneral-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkIOLegacy-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkIOXML-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkIOGeometry-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtkIOXMLParser-9.3.so.1
    ${VTK_LIBRARY_DIR}/libvtksys-9.3.so.1
    -lpthread
    -ldl
)

find_path(VTK_INCLUDE_DIR "vtkSmartPointer.h" 
    PATHS "/usr/include/vtk-9.3" "/usr/include/vtk9" "/usr/include")

include_directories(${VTK_INCLUDE_DIR})
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
    ${VTK_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

install(TARGETS umcfUTCIpostprocess DESTINATION bin)
