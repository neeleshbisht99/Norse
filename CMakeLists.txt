CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(Norse)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/NorseNet/lib)

INCLUDE_DIRECTORIES(./NorseNet/include)

SET(SOURCE_FILES
        NorseNet/include/genome.h
        NorseNet/sources/genome.cpp
        NorseNet/include/node.h
        NorseNet/sources/node.cpp
        NorseNet/include/utils.h
        NorseNet/sources/utils.cpp
        NorseNet/include/individual.h
        NorseNet/sources/individual.cpp
        NorseNet/include/norse.h
        NorseNet/sources/norse.cpp
        NorseNet/include/args.h
        NorseNet/sources/args.cpp)

ADD_LIBRARY(norse SHARED ${SOURCE_FILES})

FIND_PACKAGE(Threads REQUIRED)
TARGET_LINK_LIBRARIES(norse Threads::Threads)

INSTALL(TARGETS norse
        LIBRARY DESTINATION lib)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/NorseNet/include/args.h DESTINATION include)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/NorseNet/include/genome.h DESTINATION include)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/NorseNet/include/individual.h DESTINATION include)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/NorseNet/include/norse.h DESTINATION include)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/NorseNet/include/node.h DESTINATION include)
INSTALL(FILES ${PROJECT_SOURCE_DIR}/NorseNet/include/utils.h DESTINATION include)
