cmake_minimum_required(VERSION 3.26)
project(mpi_test)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
set(SRC common.cpp)
set(TARGETS gemv_row gemv_col gemv_block)
add_executable(gemv_row ${SRC} gemv_rows.cpp)
add_executable(gemv_col ${SRC} gemv_cols.cpp)
add_executable(gemv_block ${SRC} gemv_block.cpp)

foreach (target IN LISTS TARGETS)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
endforeach ()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
