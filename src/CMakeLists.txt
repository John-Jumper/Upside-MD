cmake_minimum_required(VERSION 2.8)
project(upside CXX C)

set (CMAKE_MODULE_PATH "../cmake;${CMAKE_MODULE_PATH}")
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Threads)

set(ARCH "native" CACHE STRING "architecture to use for -march flag to compiler")

set(DEBUG_FLAGS "-g")

set (ALL_PHASE_FLAGS "") # "-fsanitize=address")
set (DEFINES "-DR123_NO_SINCOS -Drestrict=__restrict__ -DNDEBUG")
set (OPT     "-O3 -ffast-math -fno-omit-frame-pointer -march=${ARCH}")
set (WARN    "-pedantic -Wall -Wno-strict-overflow -Wno-unused-function -Wno-unknown-pragmas")

set (CMAKE_CXX_FLAGS "${ALL_PHASE_FLAGS} ${CMAKE_CXX_FLAGS} ${DEBUG_FLAGS} -std=c++11 ${DEFINES} ${OPT} ${WARN}" )
set (CMAKE_LD_FLAGS  "${ALL_PHASE_FLAGS} ${CMAKE_LD_FLAGS}" )
include_directories(SYSTEM "include")

set (ENGINE_SRC 
    deriv_engine.cpp 
    task_graph.cpp
    nn.cpp
    main.cpp
    environment.cpp
    hbond.cpp 
    rotamer.cpp
    placement.cpp
    rama_map_pot.cpp
    spline.cpp
    sidechain_radial.cpp
    backbone_steric.cpp 
    bonds.cpp 
    eig.cpp 
    membrane_potential.cpp
    timing.cpp 
    thermostat.cpp
    h5_support.cpp 
    state_logger.cpp
    monte_carlo_sampler.cpp)

add_executable (upside ${ENGINE_SRC})

INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})
target_link_libraries(upside stdc++
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_library(upside_calculation SHARED
    engine_c_library.cpp
    ${ENGINE_SRC})

set_target_properties(upside_calculation PROPERTIES
    COMPILE_FLAGS "-DPARAM_DERIV"
    OUTPUT_NAME   "upside")

target_link_libraries(upside_calculation stdc++
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT})

add_executable(compute_rotamer_centers generate_from_rotamer.cpp compute_rotamer_centers.cpp h5_support.cpp)
target_link_libraries(compute_rotamer_centers stdc++ m ${HDF5_LIBRARIES})
set_target_properties(compute_rotamer_centers PROPERTIES EXCLUDE_FROM_ALL 1)

add_executable(compute_rotamer_pos generate_from_rotamer.cpp compute_rotamer_pos.cpp h5_support.cpp)
target_link_libraries(compute_rotamer_pos stdc++ m ${HDF5_LIBRARIES})
set_target_properties(compute_rotamer_pos PROPERTIES EXCLUDE_FROM_ALL 1)
