cmake_minimum_required(VERSION 3.20)
SET(CMAKE_C_COMPILER "C:/mingw64_posix/bin/gcc.exe")
SET(CMAKE_CXX_COMPILER "C:/mingw64_posix/bin/g++.exe")
project(OpenCLProject)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

include_directories($ENV{INCLUDE})
    #${PROJECT_SOURCE_DIR}/clFramework/)

link_directories($ENV{LIB}) 
link_libraries(OpenCL)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
	
#add_executable(enumOpenCL enumOpenCL.cpp)
add_executable(vectorAddOpenCL vectorAddOpenCL.cpp)
add_executable(matrixAddOpenCL matrixAddOpenCL.cpp)
add_executable(matrixMulOpenCL matrixMulOpenCL.cpp)

