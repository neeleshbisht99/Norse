CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(NorseNet)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

INCLUDE_DIRECTORIES(../NorseNet/include)

SET(SOURCE_FILES
        ../NorseNet/include/genome.h
        ../NorseNet/sources/genome.cpp
        ../NorseNet/include/node.h
        ../NorseNet/sources/node.cpp
        ../NorseNet/include/utils.h
        ../NorseNet/sources/utils.cpp
        ../NorseNet/include/individual.h
        ../NorseNet/sources/individual.cpp
        ../NorseNet/include/norse.h
        ../NorseNet/sources/norse.cpp
        ../NorseNet/include/args.h
        ../NorseNet/sources/args.cpp
        NorseNet.cpp)

FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(pybind11 REQUIRED)

pybind11_add_module(NorseNet ${SOURCE_FILES})
