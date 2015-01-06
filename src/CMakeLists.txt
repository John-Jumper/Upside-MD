cmake_minimum_required(VERSION 2.8)
project(upside CXX)

set (CMAKE_MODULE_PATH "../cmake;${CMAKE_MODULE_PATH}")
find_package(HDF5 REQUIRED COMPONENTS C HL)

set(ARCH "native" CACHE STRING "architecture to use for -march flag to compiler")

if(${OLD_DWARF_FORMAT})
    set(DEBUG_FLAGS "-gdwarf-3")
else()
    set(DEBUG_FLAGS "-g")
endif()

set (OMP_FLAGS "") # "-fopenmp")
set (DEFINES "-DR123_NO_SINCOS -Drestrict=__restrict__") # -DNDEBUG")
set (OPT     " -O3 -ffast-math -fno-finite-math-only -march=${ARCH}")
set (WARN    "-pedantic -Wall -Wno-strict-overflow -Wno-unused-function -Wno-unknown-pragmas")

set (CMAKE_CXX_FLAGS "${OMP_FLAGS} ${CMAKE_CXX_FLAGS} ${DEBUG_FLAGS} -std=c++11 ${DEFINES} ${OPT} ${WARN}" )
set (CMAKE_LD_FLAGS  "${OMP_FLAGS} ${CMAKE_LD_FLAGS}" )
include_directories(SYSTEM "include")

add_executable (upside
    dihedral_range.cpp
    backbone_dependent_sidechain.cpp
    sidechain_radial.cpp
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
    steric.cpp
    main.cpp 
    thermostat.cpp
    string.cpp)

INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})
target_link_libraries(upside stdc++ ${HDF5_LIBRARIES})

add_library(sidechain_interaction SHARED sidechain_interaction.cpp)
target_link_libraries(sidechain_interaction stdc++)

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_custom_target(support ALL COMMENT "Creating supporting executables")

macro(python_executable TARGET FILENAME)
    string(REGEX REPLACE "\\.py$" "" PYTHON_NEW_NAME ${FILENAME})
    add_custom_command(
	OUTPUT   ${PYTHON_NEW_NAME}
	COMMAND  ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_NEW_NAME}
	DEPENDS  ${FILENAME})
    add_custom_target(${PYTHON_NEW_NAME}_PYTHON DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_NEW_NAME})
    add_dependencies(${TARGET} ${PYTHON_NEW_NAME}_PYTHON)
endmacro(python_executable)

python_executable(support upside_config.py)
python_executable(support extract_vtf.py)
python_executable(support PDB_to_initial_structure.py)
python_executable(support P_of_r.py)
python_executable(support attr_overview.py)

add_executable(rotamer_sidechain generate_from_rotamer.cpp test_rotamer.cpp)
target_link_libraries(rotamer_sidechain stdc++ m ${HDF5_LIBRARIES})
