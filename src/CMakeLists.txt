cmake_minimum_required(VERSION 2.8)
project(upside CXX)

set (CMAKE_MODULE_PATH "../cmake;${CMAKE_MODULE_PATH}")
find_package(HDF5 REQUIRED)
# find_package(VTF REQUIRED)

set (DEFINES "-DR123_NO_SINCOS -Drestrict=__restrict__ -DNDEBUG")
set (OPT     " -O3 -ffast-math -march=native")
set (WARN    "-pedantic -Wall -Wno-strict-overflow -Wno-unused-function")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 ${DEFINES} ${OPT} ${WARN}" )
set (CMAKE_LD_FLAGS  "${CMAKE_LD_FLAGS}" )
include_directories(SYSTEM "include")

add_executable (upside
    sidechain.cpp 
    h5_support.cpp 
    force.cpp 
    nonbond.cpp 
    md.cpp 
    hmm.cpp 
    eig.cpp 
    hbond.cpp 
    bonds.cpp 
    timing.cpp 
    main.cpp 
    thermostat.cpp)

INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIR})
target_link_libraries(upside stdc++ ${HDF5_LIBRARIES} hdf5_hl) # ${VTK_LIBRARIES})

add_library(sidechain_interaction SHARED sidechain_interaction.cpp)
target_link_libraries(sidechain_interaction stdc++)

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
# add_library(analysis SHARED analysis.cpp hbond.cpp)
# target_link_libraries(analysis stdc++)
# set_target_properties(analysis PROPERTIES PREFIX "")
